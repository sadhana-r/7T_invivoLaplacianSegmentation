#!/bin/bash

source /data/sadhanar/env_DL_sadhana/bin/activate

export nnUNet_raw_data_base="/data/sadhanar/7TSegmentationInvivo/nnUNet_raw_data_base"
export nnUNet_preprocessed="/data/sadhanar/7TSegmentationInvivo/nnUNet_preprocessed"
export RESULTS_FOLDER="/data/sadhanar/7TSegmentationInvivo/nnUNet_trained_models"

CUDA_VISIBLE_DEVICES=3 nnUNet_predict -i '/data/sadhanar/7TSegmentationInvivo/7T_forinference/data' -o '/data/sadhanar/7TSegmentationInvivo/7T_forinference/baseline_predictions' -t 700
