#!/bin/bash

source /data/sadhanar/env_DL_sadhana/bin/activate

export nnUNet_raw_data_base="/data/sadhanar/7TSegmentationInvivo/nnUNet_raw_data_base"
export nnUNet_preprocessed="/data/sadhanar/7TSegmentationInvivo/nnUNet_preprocessed"
export RESULTS_FOLDER="/data/sadhanar/7TSegmentationInvivo/nnUNet_trained_models"

<<"BASELINE"
#nnUNet_plan_and_preprocess -t 700 --verify_dataset_integrity

for fold in 1 2 3; do # 0
       echo $fold
       CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2 700 $fold  --npz
done
BASELINE

# To train the Laplacian version of nn-UNet, copy the files in nnunet_modified_scipts into their respective locations 
# in the nn-UNet directory structure. The list with locations is provided in nnunetmodifications_README.md.

#nnUNet_plan_and_preprocess -t 701 --verify_dataset_integrity

CUDA_VISIBLE_DEVICES=3 nnUNet_train 3d_fullres nnUNetTrainerV2_SORseg_7TInvivo 701 'all'



