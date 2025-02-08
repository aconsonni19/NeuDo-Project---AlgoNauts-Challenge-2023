import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr

### This code is



### Select computation device, if cuda is available,
### the algorithm will perform the computations on the GPU, otherwise
### the CPU will be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Path to the root folter of the dataset
root_data_dir = "..\\..\\Dataset"

### Folder where the

