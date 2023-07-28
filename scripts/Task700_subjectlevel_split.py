from batchgenerators.utilities.file_and_folder_operations import *
import os
import numpy as np
from collections import OrderedDict

## This script creates a custom 4-fold split of the data to
## make sure that the same subject isn't included in the training and validation patches. 

data_split_custom = []
subjects = [['20200908x1133','20201022x1146','20201201x1636'],['20201208x1230','20201208x1405','20210722x1003'],
['20210921x1131','20211007x1641','20211021x1115'],['20220106x1012','20220405x1035','20220421x0935']]


for fold in range(4):

    idx = [i for i in range(4) if i != fold]

    train_list = []
    train_subjects_list = []
    train_subjects_list += (subjects[i] for i in idx)
    for train_subject in np.concatenate(train_subjects_list):
        for side in ['R','L']:
            for patch in ['0','1','2']:
                filename = train_subject + "_3TSegTo7TDeformed_" + side + "_" +patch 
                train_list.append(filename)

    val_list = []
    for val_subject in subjects[fold]:
        for side in ['R','L']:
            for patch in ['0','1','2']:
                filename = val_subject + "_3TSegTo7TDeformed_" + side + "_" +patch
                val_list.append(filename)

    data_split_custom.append(OrderedDict([('train', train_list),
                      ('val', val_list)]))

print(data_split_custom[0])

save_pickle(data_split_custom,'/data/sadhanar/7TSegmentationInvivo/nnUNet_preprocessed/Task700_7T_DeformedSeg/splits_final.pkl' )