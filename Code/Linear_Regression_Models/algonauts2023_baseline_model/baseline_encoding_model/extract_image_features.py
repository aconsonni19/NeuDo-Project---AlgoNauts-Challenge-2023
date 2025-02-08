import argparse

import joblib
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA


# =============================================================================
# Definition of AlexNet model
# =============================================================================
class AlexNet(nn.Module):
	# Lists of AlexNet convolutional and fully connected layers

	def __init__(self):
		"""Select the desired layers and create the model."""
		super(AlexNet, self).__init__()
		self.select_cov = ['maxpool1', 'maxpool2', 'ReLU3', 'ReLU4', 'maxpool5']
		self.select_fully_connected = ['ReLU6' , 'ReLU7', 'fc8']
		self.feat_list = self.select_cov + self.select_fully_connected
		self.alex_feats = models.alexnet(pretrained=True).features
		self.alex_classifier = models.alexnet(pretrained=True).classifier
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

	def forward(self, x):
		"""Extract the feature maps."""
		conv_layers = ['conv1', 'ReLU1', 'maxpool1', 'conv2', 'ReLU2', 'maxpool2',
					   'conv3', 'ReLU3', 'conv4', 'ReLU4', 'conv5', 'ReLU5', 'maxpool5']
		fully_connected_layers = ['Dropout6', 'fc6', 'ReLU6', 'Dropout7', 'fc7',
								  'ReLU7', 'fc8']
		features = []
		for name, layer in self.alex_feats._modules.items():
			x = layer(x)
			if conv_layers[int(name)] in self.feat_list:
				features.append(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		for name, layer in self.alex_classifier._modules.items():
			x = layer(x)
			if fully_connected_layers[int(name)] in self.feat_list:
				features.append(x)
		return features



def extract_features(sub, project_dir, output_dir):
	"""
	Extract the training/testing feature maps using a pretrained AlexNet, and
	downsample them to 1000 PCA components

	:param sub: int
		subject used
	:param project_dir:
		project_dir
	:param output_dir:
		output_dir
	:return:
		writes the downsampled features maps (.npy files) in the output directory
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print('>>> Algonauts 2023 extract image features <<<')
	print(f'Project directory: {project_dir}')
	print(f'Output directory: {output_dir}')
	print(f'Subject used: {sub}')
	# Set random seed for reproducible results
	seed = 20200220
	torch.manual_seed(seed)
	model = AlexNet()
	if torch.cuda.is_available():
		print("Utilizing GPU")
		model.cuda()
	model.eval()
	# =============================================================================
	# Define the image preprocessing
	# =============================================================================
	centre_crop = trn.Compose([
		trn.Resize((224, 224)),
		trn.ToTensor(),
		trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	# ================================================================================================
	# Load the images and extract the corresponding feature maps; the load wil be performed in batches
	# ================================================================================================
	# Extract the feature maps of (1) training images and (2) test images
	# Image directories
	img_set_dir = os.path.join(project_dir, 'Dataset')
	splits_parent = ['train_data', 'test_data']
	splits = ['training_split', 'test_split']
	splits_child = ['training_images', 'test_images']
	fmaps_train = []
	fmaps_test = []
	batch_size = 1024
	batch_files = [fmaps_train, fmaps_test]
	for s in range(len(splits_parent)):
		image_list = os.listdir(os.path.join(img_set_dir, splits_parent[s], 'subj' +
											 format(sub, '02'), splits[s], splits_child[s]))
		image_list.sort()
		print("Extracing features from " + splits_parent[s] + f' in batches of {batch_size}')
		for i in range(0, len(image_list), batch_size): ### Process batch by batch
			batch_images = image_list[i:i + batch_size]
			batch_features = []
			for image in tqdm(batch_images):
				img = Image.open(os.path.join(img_set_dir, splits_parent[s], 'subj' +
											  format(sub, '02'), splits[s], splits_child[s],
											  image)).convert('RGB')
				input_img = V(centre_crop(img).unsqueeze(0))
				if torch.cuda.is_available():
					input_img = input_img.cuda()
				x = model.forward(input_img)

				img_feats =  np.concatenate([feat.data.cpu().numpy().flatten() for feat in x])
				batch_features.append(img_feats)

			batch_features = np.asarray(batch_features)

			save_dir = os.path.join(output_dir, 'batch_features_maps', splits_parent[s], 'subj' + format(sub, '02'))
			if os.path.isdir(save_dir) == False:
				os.makedirs(save_dir)
			batch_file = os.path.join(save_dir, splits_parent[s] + f'_batch_features_{int(i/batch_size)}.npy')
			np.save(batch_file, batch_features)
			batch_files[s].append(batch_file)

	# =============================================================================
	# Apply PCA on the training images feature maps
	# =============================================================================
	# The standardization and PCA statistics computed on the training images
	# feature maps are also applied to the test images feature maps
	print("Applying PCA on the training image feature maps...")
	# Standardize the data
	sc = StandardScaler()

	### Compute mean e standard deviation incrementally
	print("Standardizing the training data...")
	for batch_file in tqdm(fmaps_train):
		batch_data = np.load(batch_file)
		sc.partial_fit(batch_data)

	### Saving the computed scaler:
	Scaler_model_save_dir = os.path.join(output_dir, 'PCA_scalers')
	if os.path.isdir(Scaler_model_save_dir) == False:
		os.makedirs(Scaler_model_save_dir)
	joblib.dump(sc, os.path.join(Scaler_model_save_dir, f'training_scaler_subj_{sub}.pkl'))

	# Apply PCA using IncrementalPCA with processed data chunks
	n_components = 100 # Reduce dimensionality to 100 components
	pca = IncrementalPCA(n_components=n_components)

	print("Applying PCA to all training batches...")
	### Apply standardization and PCA in batches
	for batch_file in tqdm(fmaps_train):
		batch_data = np.load(batch_file)
		batch_data = sc.transform(batch_data) # Standardized batch
		pca.partial_fit(batch_data) # Fit PCA incrementally

	# Save PCA model:
	PCA_model_save_dir = os.path.join(output_dir, 'PCA_models')
	if os.path.isdir(PCA_model_save_dir) == False:
		os.makedirs(PCA_model_save_dir)
	joblib.dump(pca, os.path.join(PCA_model_save_dir, f'train_pca_subj_{sub}.pkl'))

	save_dir = os.path.join(output_dir, 'dnn_feature_maps', 'subj' + format(sub, '02'), 'train_data')
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	### Transformation with PCA on each batch:
	print("Saving the downasmpled features map...")
	i = 0
	for batch_file in tqdm(fmaps_train):
		batch_data = np.load(batch_file)
		batch_data = sc.transform(batch_data) # Standardize
		batch_data_pca = pca.transform(batch_data) # Apply PCA
		np.save(os.path.join(save_dir, f'train_data_batch_features_pca_{i}'), batch_data_pca)
		i += 1
	del fmaps_train ### Free computing resources

	# =============================================================================
	# Apply PCA on the test images feature maps
	# =============================================================================
	print("Standardizing and applying PCA on the test images feature maps...")
	save_dir = os.path.join(output_dir, 'dnn_feature_maps', 'subj' + format(sub, '02'), 'test_data')
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	# Standardize the data wit the previously saved scaler and PCA mdoels:
	sc = joblib.load(os.path.join(Scaler_model_save_dir, f'training_scaler_subj_{sub}.pkl'))
	pca = joblib.load(os.path.join(PCA_model_save_dir, f'train_pca_subj_{sub}.pkl'))

	i = 0
	for batch_file in tqdm(fmaps_test):
		batch_data = np.load(batch_file)
		batch_data = sc.transform(batch_data) # Standardize Batch
		batch_data_pca = pca.transform(batch_data) # Apply PCA
		np.save(os.path.join(save_dir, f'test_data_batch_features_pca_{i}'), batch_data_pca)
		i += 1
	del fmaps_test ### Free computing resources