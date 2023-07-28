import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
import nibabel as nib
import os

def read_nifti(filepath_image):

    img = nib.load(filepath_image)
    image_data = img.get_fdata()

    return image_data, img


def save_nifti(image, filepath_name, img_obj):

    img = nib.Nifti1Image(image, img_obj.affine, header=img_obj.header)
    nib.save(img, filepath_name)


if __name__ == '__main__':
    """
    nnU-Net was originally built for 3D images. It is also strongest when applied to 3D segmentation problems because a
    large proportion of its design choices were built with 3D in mind. Also note that many 2D segmentation problems,
    especially in the non-biomedical domain, may benefit from pretrained network architectures which nnU-Net does not
    support.
    Still, there is certainly a need for an out of the box segmentation solution for 2D segmentation problems. And
    also on 2D segmentation tasks nnU-Net cam perform extremely well! We have, for example, won a 2D task in the cell
    tracking challenge with nnU-Net (see our Nature Methods paper) and we have also successfully applied nnU-Net to
    histopathological segmentation problems.
    Working with 2D data in nnU-Net requires a small workaround in the creation of the dataset. Essentially, all images
    must be converted to pseudo 3D images (so an image with shape (X, Y) needs to be converted to an image with shape
    (1, X, Y). The resulting image must be saved in nifti format. Hereby it is important to set the spacing of the
    first axis (the one with shape 1) to a value larger than the others. If you are working with niftis anyways, then
    doing this should be easy for you. This example here is intended for demonstrating how nnU-Net can be used with
    'regular' 2D images. We selected the massachusetts road segmentation dataset for this because it can be obtained
    easily, it comes with a good amount of training cases but is still not too large to be difficult to handle.
    """

    base = '/data/sadhanar/7TSegmentationInvivo/'
    # this folder should have the training and testing subfolders

    # now start the conversion to nnU-Net:
    task_name = 'Task700_7T_DeformedSeg'
    target_base = join(base,'nnUNet_raw_data_base/nnUNet_raw_data',task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    data_dir = join(base, 'patches')
    for subject in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject)
        if os.path.isdir(subject_dir):
            print(subject_dir)
            for side in ['R','L']:
                for patch in ['0','1','2']:
                    patch_dir = join(subject_dir,side,patch,subject)
                    print(patch_dir)
                    input_image_file = patch_dir + '_' + side + '_T1w_7T_Preproc_' + patch + '.nii.gz'
                    
                    input_seg_file = patch_dir +'_' + side + '_3TSegTo7TDeformed_' + patch + '.nii.gz'

                    output_image_file = join(target_imagesTr, subject)  # do not specify a file ending! This will be done for you
                    output_image_file = output_image_file + "_3TSegTo7TDeformed_" + side + "_"+patch + "_0000.nii.gz" 

                    output_seg_file = join(target_labelsTr, subject)  # do not specify a file ending! This will be done for you
                    output_seg_file = output_seg_file + "_3TSegTo7TDeformed_" + side + "_"+patch + ".nii.gz" 

                    # read image and save it
                    image_data, img_obj = read_nifti(input_image_file)
                    save_nifti(image_data, output_image_file, img_obj)

                    #Read seg and save it
                    image_data, img_obj = read_nifti(input_seg_file)
                    save_nifti(image_data, output_seg_file, img_obj)

 # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('MRI',),
                          labels={0: 'None', 1: 'CSF', 2: 'GM', 3: 'WM', 4: 'Other', 5: 'Misc', 6: 'Label 6'}, dataset_name=task_name, license='hands off!')
