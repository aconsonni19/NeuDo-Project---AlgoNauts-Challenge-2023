import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from joblib.externals.cloudpickle import subimport
from torchvision.models.vision_transformer import vit_b_16
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from vit_train_and_test import extract_vit_features, train_cnn_models, test_mlp_encoding, mlp_plot_roi_correlation, \
    mlp_accuarcy_plot, mlp_accuracy_plot_fsavarage

### Local dirs
project_dir = "../../"
output_dir = "../../Outputs"
subjects = [7, 8]
vit_model_output_dir = os.path.join(output_dir, 'ViT_based')


for sub in subjects:
    extract_vit_features(sub, "vit_base_patch16_224", project_dir, vit_model_output_dir)
    train_cnn_models(sub, project_dir, vit_model_output_dir, trials=5)
    subdirs = [x[1] for x in os.walk(vit_model_output_dir)][0]
    subdirs.remove("vit_features")
    subdirs.remove("PCA_models")
    subdirs.remove("PCA_scalers")
    subdirs.remove("dnn_feature_maps")# Remove unwanted dir
    for subdir in subdirs:
        print(f'Analyzing subdir {subdir}')
        model_correlations = test_mlp_encoding(sub, project_dir, os.path.join(vit_model_output_dir, subdir))
        mlp_plot_roi_correlation(sub, project_dir, os.path.join(vit_model_output_dir, subdir),
                                 model_correlations)
        mlp_accuarcy_plot(sub, project_dir, os.path.join(vit_model_output_dir, subdir))
        mlp_accuracy_plot_fsavarage(sub, project_dir, os.path.join(vit_model_output_dir, subdir))

