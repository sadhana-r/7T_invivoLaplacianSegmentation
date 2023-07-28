import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
import nibabel as nib
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

#Convert segmentation to a one hot encoded image
def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result

#Sigmoid filters to convert laplacian to a segmentation
def doublesigmoid_threshold(data, lower_lim, upper_lim):

    steepness = 10

    lower_thresh = 1/(1 + torch.exp(-steepness*(data - lower_lim)))
    upper_thresh = 1/(1 + torch.exp(steepness*(data - upper_lim)))

    #Combine both filters
    output = torch.mul(lower_thresh,upper_thresh)
    output = output.squeeze()

    return output


def convert_laplacian_toseg(data, thresholds = torch.tensor([-0.3,0, 0.25,0.5,0.75,0.95])):

    #thresholds = torch.tensor([-0.3,0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95]) #For fine version

    #Create output image
    result = torch.zeros((data.shape[0],len(thresholds), *data.shape[2:]), dtype=data.dtype)

    #For each set of threshold values, generate the sigmoid filters to generate the
    #correspondig "one-hot encoded" channel
    for i, l in enumerate(thresholds):

        output = doublesigmoid_threshold(data, l,l+0.1)
        result[:,i] = output

    return result

#Differential Laplacian Solver
class SuccessiveOverRelaxation_opt(nn.Module):

    def __init__(self,source, sink, domain, gt = False, threshold = 0.05, w = 1.5, max_iterations = 120):
        super(SuccessiveOverRelaxation_opt, self).__init__()

        """
        w is the over-relaxation parameter
        """
        self.wopt = w
        self.thresh = threshold
        self.source = source
        self.sink = sink
        self.domain = domain
        self.max_iterations = max_iterations
        self.gt = gt

    #Convert 3D index to the corresponding index in flattened space
    def ravel_index(self,index, shape):

        [dim1,dim2,dim3] = shape
        out = []
        for i in range(index.shape[1]):
            out.append((dim2*dim3)*index[0,i] +
            dim3*index[1,i] + index[2,i])

        return torch.stack(out)

    def forward(self, image):

        """
        image should be of size NxCxHxDxW where C is 1 and N can be greater than 1
        solver iterates through each image in the batch
        """

        #Extract input image dimensions
        bs,c,h,w,d = image.shape

        #Initialize solver. WM is source, CSF is sink, GM is the region over which
        # Laplacian is solved (initialize to 0.5)
        if not self.gt:
            init = 0.5*image[:,0,:] + 1*image[:,1,:] + 0*image[:,2,:] # These channel numbers are wrong - shouldn't be hard coded

            #This should be the gm probability map, label 1
            mask_gm = image[:,0,:].unsqueeze(1)

        else:
            # Special case ground truth
            init = 0.5*image[:,self.domain,:] + 1*image[:,self.source,:] + 0*image[:,self.sink,:]

            #This should be the gm probability map, label 1
            mask_gm = image[:,self.domain,:].unsqueeze(1)

        #Maintain the channel dimension
        init = init.unsqueeze(1)

        #Calculate the optimal over-relaxation parameter given the image size
        min_dim = torch.min(torch.tensor(init.shape)).type(torch.float32)
        self.wopt = 2/(1+(np.pi/min_dim))

        #Black and red coordinate masks
        xx,yy,zz = torch.meshgrid(torch.arange(0,h), torch.arange(0,w), torch.arange(0,d))
        #, indexing = 'ij'
        coords = xx + yy + zz
        coords = coords[None, None,:]
        coords = coords.repeat([bs,1,1,1,1])

        black = torch.zeros(coords.shape)
        red = torch.zeros(coords.shape)

        #Create the checkboard pattern masks based on the index of the coordinates
        red[(torch.fmod(coords,2) == 0)] = 1
        black[(torch.fmod(coords,2) == 1)] = 1

        #Initialize the iteration counter
        iterations = 0

        #Create the kernel for the SOR convolutions - function of the w_opt parameter
        #Takes the weighted sum of neighboring voxels
        w_xy = self.wopt/6
        weight_kernel = torch.tensor([[[0,0,0],[0,w_xy,0],[0,0,0]],[[0,w_xy,0],[w_xy,1-self.wopt,w_xy],[0,w_xy,0]],[[0,0,0],[0,w_xy,0],[0,0,0]]])
        weight_kernel = weight_kernel[None, None, :]

        while(iterations < self.max_iterations):

            #Apply first convolutional layer
            half_step =  F.conv3d(init, weight_kernel, padding = 'same')

            #Combine black step with previous red step
            half_step = half_step*black + init*red

            #Mask to retain only gray matter pixels
            half_step = half_step*mask_gm + init*(torch.ones(mask_gm.shape) - mask_gm)

            #Apply the second convolutional layer
            full_step = F.conv3d(half_step, weight_kernel, padding = 'same')

            # Combine red solution and black solution before masking)
            full_step = full_step*red + half_step*black

            #Only keep SOR solution within gm
            full_step = full_step*mask_gm + init*(torch.ones(mask_gm.shape) - mask_gm)

            init = full_step

            iterations += 1

        return init, iterations

def read_nifti(filepath_image):

    img = nib.load(filepath_image)
    image_data = img.get_fdata().astype(np.float32)

    return image_data, img


def save_nifti(image, filepath_name, img_obj):

    img = nib.Nifti1Image(image, img_obj.affine, header=img_obj.header)
    nib.save(img, filepath_name)


if __name__ == '__main__':

    base = '/data/sadhanar/7TSegmentationInvivo/'

    # now start the conversion to nnU-Net:
    task_name = 'Task701_7T_ManualSeg_Laplacian'

    target_base = join(base,'nnUNet_raw_data_base/nnUNet_raw_data',task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    #For saving Laplacian segmentations
    maybe_mkdir_p(join(base, 'manualpatches_LaplacianSeg'))

    #Input data is the manual patches where the sulci has been cleaned up
    manual_patches_dir = join(base, 'manual patches')
    data_dir = join(base, 'patches')

    #Initialize SOR solver
    sor = SuccessiveOverRelaxation_opt(source = 1, sink = 3, domain = 2, gt = True) # White matter is label 1, CSF is label 3

    for subject in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject)
        if os.path.isdir(subject_dir):
            print(subject_dir)
            for side in ['R','L']:
                for patch in ['0','1','2']:
                    patch_dir = join(subject_dir,side,patch,subject)
                    input_image_file = patch_dir + '_' + side + '_T1w_7T_Preproc_' + patch + '.nii.gz'
                    print(input_image_file)

                    seg_file_pattern = manual_patches_dir + '/' + subject + '_' + side + '*' + patch + '*.nii.gz'
                    input_seg_file = glob.glob(seg_file_pattern) # For two patches, there are 4 versions (from Inter-rater reliability analysis - currently just using the first file)

                    #Not all segmentations have been edited. So need to have this check for now
                    if input_seg_file != []:
                        
                        print(len(input_seg_file))
                        output_image_file = join(target_imagesTr, subject)  # do not specify a file ending! This will be done for you
                        output_image_file = output_image_file + "_ManualSeg_" + side + "_"+patch + "_0000.nii.gz" 

                        output_seg_file = join(target_labelsTr, subject)  # do not specify a file ending! This will be done for you
                        output_seg_file = output_seg_file + "_ManualSeg_" + side + "_"+patch + ".nii.gz" 

                        #Read seg 
                        print(input_seg_file[0])
                        seg_data, img_obj = read_nifti(input_seg_file[0])

                        #Need to generate the ground truth Laplacian corresponding to the ground truth segmentation
                        # and save it as a second channel (*_0001.nii.gz) in the training image file.

                        #Convert to one hot encoded image - to resemble probability map output by network
                        seg_data_one_hot = to_one_hot(seg_data)
                        seg_data_one_hot = torch.tensor(seg_data_one_hot[None,:])

                        output_laplacian_file = join(target_imagesTr, subject)  # do not specify a file ending! This will be done for you
                        output_laplacian_file = output_laplacian_file + "_ManualSeg_" + side + "_"+patch + "_0001.nii.gz" 

                        #Solve laplacian solution

                        #Note: Some patches are mostly empty - so they dont contain WM, GM and CSF labels required for
                        # solving the laplacian solution. In such cases, ignore the patch 

                        # If atleast the three main labels are included in the segmentation, save the patch
                        if seg_data_one_hot.shape[1] > 3:
                            laplace_sol,_ = sor(seg_data_one_hot)

                            #Save Laplacian solution to file
                            output_laplace = laplace_sol[0,0,:].numpy()
                            save_nifti(output_laplace, output_laplacian_file, img_obj)

                            # This is for dice evaluation - not required for training
                            #Output laplacian segmentation filename
                            output_laplacianseg_file = join(base, 'manualpatches_LaplacianSeg', subject)
                            output_laplacianseg_file = output_laplacianseg_file + "_Laplacian_" + side + "_"+patch + '.nii.gz'

                            #Convert the laplacian solution to a segmentation - 'coarse' version has 5 labels. Adjust thresholds for in vivo
                            laplace_seg = convert_laplacian_toseg(laplace_sol, thresholds = torch.tensor([-0.3,0, 0.25,0.5,1]))
                            laplace_seg = laplace_seg.argmax(1)

                            #Mask the laplcian seg by the ground truth segmentation
                            laplace_seg = laplace_seg[0,:]

                            laplace_seg = laplace_seg.numpy()
                            save_nifti(laplace_seg, output_laplacianseg_file, img_obj)

                            #Only save the segmentation and image if the Laplacian step is possible
                            save_nifti(seg_data.astype(np.int32), output_seg_file, img_obj)

                            # read image and save it
                            image_data, img_obj = read_nifti(input_image_file)
                            save_nifti(image_data, output_image_file, img_obj)
                        else:
                            print("Patch doesn't contain enough labels to compute Laplacian")

 # finally we can call the utility for generating a dataset.json. Important to include second channel here and specify 'noNorm' so the Laplacian doesn't get normalized. 
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('MRI','noNorm'),
                          labels={0: 'None', 1: 'CSF', 2: 'GM', 3: 'WM', 4: 'Other', 5: 'Misc', 6: 'Label 6'}, dataset_name=task_name, license='hands off!')
