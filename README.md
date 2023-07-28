# 7T_invivoLaplacianSegmentation
Scripts for training the modified nnU-Net + Laplacian model on 7T MRI for cortical segmentation. All files/data are located on the lambda cluster. 

## Setup

To set up the modified version of /path/to/nnUNet/, first install nnU-Net V1 as described in the documentation: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1. I have modified various scripts to incorporate the Laplacian solver in the nnU-Net framework. To train the modified network, copy the files in nnunet_modified_scipts to their respective locations within your nnU-Net installation folder. 

The location of the different files is as follows:

1) Network Trainer:
   - /path/to/nnUNet/nnunet/training/network_training/nnUNetTrainerV2_SORseg_7TInvivo.py 

3) Network Architecture:
   - /path/to/nnUNet/nnUNet/nnunet/network_architecture/neural_network_SOR.py
   - /path/to/nnUNet/nnUNet/nnunet/network_architecture/generic_UNet_SOR.py

6) Data Augmentation:
   - /path/to/nnUNet/nnUNet/nnunet/training/data_augmentation/move_datachannel_to_seg.py     
   - /path/to/nnUNet/nnUNet/nnunet/training/data_augmentation/data_augmentation/data_augmentation_moreDA_SOR.py       
   - /path/to/nnUNet//nnUNet/nnunet/training/data_augmentation/data_augmentation/downsampling_SOR.py

7) Data Loading:
   - /path/to/nnUNet/nnUNet/nnunet/training/dataloading/dataset_loading_SOR.py

9) Loss Function:
    - /path/to/nnUNet/nnUNet/nnunet/training/loss_functions/deep_supervision_7Tinvivo.py

11) Utilities:
    - /path/to/nnUNet/nnUNet/nnunet/utilities/convert_laplacian_to_seg.py

## Preparing data for training

- `Task700_7T_deformedSeg.py`: To train the baseline model (Task 700 - without Laplacian constraints), the script assumes the input data is in the 'patches' folder with format ../patches/session/R(L)/*.nii.gz. This python script organizes the data in the format required by `Task700_7T_deformedSeg.py`: for training.
- `Task700_subjectlevel_split.py`: For running cross-validation experiments, the input training patches need to be stratified at a subject level and not patch level. I therefore created a custom 4-fold split based on subject stratification. After running `nnUNet_plan_and_preprocess -t 700 --verify_dataset_integrity`, update splits_final.pkl by running this script.
- `PrepWarpseg_forinference.py`: After copying the preprocessed WarpSeg folder from the pmacs cluster to the lambda machine, this script prepares the input data into the format required for running nnU-Net inference. The organized data is stored in ../7T_forinference. 
- `run_training.sh`: Executes patch-based nnU-Net training. I have trained the baseline model in a 4-fold cross validation setting. The results are stored in /data/sadhanar/nnUNet_trained_models/.../Task700..
- `run_inference_warpSeg.sh`: Script for running inference on the whole brain MRI images stored in 7T_forinfernece. I have run baseline model inference on these images. The results are stored in /data/sadhanar/baseline_predictions. 
- `Task701_7T_ManualSeg_Laplacian.py': Given the 'manual patches' folder from Box with the edited segmentations, this script prepares the input data for nnU-Net training. It generates the Laplacian solution corresponding to the ground truth segmentation as this is required for training. Note that these images are input into the network as a second 'channel (modality)' and not as a label image. 


