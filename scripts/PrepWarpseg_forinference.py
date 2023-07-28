import numpy as np
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

    base = '/data/sadhanar/7TSegmentationInvivo/'

    output_image_dir = os.path.join(base,'7T_forinference', 'data')
    output_seg_dir = os.path.join(base,'7T_forinference', 'groundtruth')

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_seg_dir, exist_ok=True)

    data_dir = os.path.join(base, 'from_pmacs','warpSeg')

    for session in os.listdir(data_dir):

        print(session)
        session_dir = os.path.join(data_dir, session, session)        
        input_image_file = session_dir +'_T1w_7T_Preproc.nii.gz'
        input_seg_file = session_dir +'_3TSegTo7TDeformed.nii.gz'

        output_image_file = os.path.join(output_image_dir, session)  # do not specify a file ending! This will be done for you
        output_image_file = output_image_file + "_T1w_7T_preproc_patchnnUnet_0000.nii.gz" 

        output_seg_file = os.path.join(output_seg_dir, session)  # do not specify a file ending! This will be done for you
        output_seg_file = output_seg_file + "_3TSegTo7TDeformed.nii.gz" 

        # read image and save it
        image_data, img_obj = read_nifti(input_image_file)
        save_nifti(image_data, output_image_file, img_obj)
    
        # read image and save it
        image_data, img_obj = read_nifti(input_seg_file)
        save_nifti(image_data, output_seg_file, img_obj)

