# 7T_invivoLaplacianSegmentation
Scripts for training the modified nnU-Net + Laplacian model on 7T MRI for cortical segmentation

# Setup

To set up the modified version of nnU-Net, first install nnU-Net V1 as described in the documentation: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1. I have modified various scripts to incorporate the Laplacian solver in the nnU-Net framework. To train the modified network, copy the files in nnunet_modified_scipts to their repsective locations within your nnU-Net installation folder. 

The location of the different files is as follows:

1) Network Trainer:
   - /path/to/nnUNet/nnunet/training/network_training/nnUNetTrainerV2_SORseg_7TInvivo.py 

3) Network Architecture:
   - /path/to/nnUNet/nnUNet/nnunet/network_architecture/neural_network_SOR.py
   - /path/to/nnUNet/nnUNet/nnunet/network_architecture/generic_UNet_SOR.py

6) Data Augmentation:
   - /path/to/nnUNet/nnUNet/nnunet/training/data_augmentation/move_datachannel_to_seg.py -       
   - /path/to/nnUNet/nnUNet/nnunet/training/data_augmentation/data_augmentation/data_augmentation_moreDA_SOR.py       
   - /path/to/nnUNet//nnUNet/nnunet/training/data_augmentation/data_augmentation/downsampling_SOR.py

7) Data Loading:
   - /path/to/nnUNet/nnUNet/nnunet/training/dataloading/dataset_loading_SOR.py

9) Loss Function:
    - /path/to/nnUNet/nnUNet/nnunet/training/loss_functions/deep_supervision_7Tinvivo.py

11) Utilities:
    - /path/to/nnUNet/nnUNet/nnunet/utilities/convert_laplacian_to_seg.py



