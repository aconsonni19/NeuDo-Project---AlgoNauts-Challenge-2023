"""Extract the training/testing feature maps using a pretrained AlexNet, and
downsample them to 1000 PCA components.

Parameters
----------
sub : int
    Used NSD subject.
project_dir_dir : str
    Project directory.

"""

import argparse
import gc
import glob
import random

import joblib
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
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
from sklearn.decomposition import IncrementalPCA
from scipy.stats import pearsonr
from copy import copy
import cortex

# =============================================================================
# Definition of AlexNet model
# =============================================================================

class AlexNet(nn.Module):
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
        # =============================================================================
        # Select the layers of interest and import the model
        # =============================================================================
        # Lists of AlexNet convolutional and fully connected layers
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

# =============================================================================
# Algorithm to extract features using AlexNet
# =============================================================================

def baseline_encoding_extract_features(sub, project_dir, output_dir):
    print('>>> Algonauts 2023 extract image features <<<')
    print(f'Project directory: {project_dir}')
    print(f'Output directory: {output_dir}')
    print(f'Subject used: {sub}')

    # Setting environment variable to allow deterministic behaviour
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.cuda.empty_cache()  # Clears unused GPU memory
    gc.collect()  # Forces Python to release memory

    # Set random seed for reproducible results
    seed = 20200220
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    model = AlexNet()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()


    # =============================================================================
    # Define the image preprocessing
    # =============================================================================
    centre_crop = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # =============================================================================
    # Load the images and extract the corresponding feature maps
    # =============================================================================
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
        print("Extracting features from " + splits_parent[s] + f' in batches of {batch_size}')

        # Extract the feature maps from batch:
        for i in range(0, len(image_list), batch_size):
            batch_images = image_list[i:i + batch_size] # Extract a subset of batch_size images
            batch_features = [] # List that will contain the batch images features

            for image in tqdm(batch_images): # Extraction of features
                img = Image.open(os.path.join(img_set_dir, splits_parent[s], 'subj' +
                                              format(sub, '02'), splits[s], splits_child[s],
                                              image)).convert('RGB') #Open image
                input_img = V(centre_crop(img).unsqueeze(0)) # Crop image to center and adds dimension for batch
                                                             # processing
                if torch.cuda.is_available(): # Moves image to GPU for processing
                    input_img = input_img.cuda()
                x = model.forward(input_img)

                # Extracts the features from the output of the model,
                # converts them to a 1D NumPy array, and appends the array to batch_features
                img_feats = np.concatenate([feat.data.cpu().numpy().flatten() for feat in x])
                batch_features.append(img_feats)

            # Convert batch features to numpy array
            batch_features = np.asarray(batch_features)

            # Save batch features
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
    print("Applying PCA on the training image features maps...")

    # Initialize scaler: the scaler used will transform the data such that
    # its distribution will have a mean of 0 and a variance of 1
    sc = StandardScaler()

    # Compute mean e standard deviation incrementally
    print("Computing mean and standard deviation for all batches...")
    for batch_file in tqdm(fmaps_train):
        batch_data = np.load(batch_file)
        sc.partial_fit(batch_data)

    # Save scaler model for future use:
    Scaler_model_save_dir = os.path.join(output_dir, 'PCA_scalers')
    if os.path.isdir(Scaler_model_save_dir) == False:
        os.makedirs(Scaler_model_save_dir)
    joblib.dump(sc, os.path.join(Scaler_model_save_dir, 'scaler_subj_' + format(sub, '02') + '.pkl'))

    # Applying Principal Component Analysis on the processed image batches
    n_components = 100  # Reduce dimensionality to 100 components
    pca = IncrementalPCA(n_components= 100) #

    print("Applying PCA to all training batches...")
    for batch_file in tqdm(fmaps_train):
        batch_data = np.load(batch_file) # Load batch data
        batch_data = sc.transform(batch_data) # Standardize batch using the scaler computed previously
        pca.partial_fit(batch_data) # Fit PCA incrementally

    # Save PCA model for future use:
    PCA_model_save_dir = os.path.join(output_dir, 'PCA_models')
    if os.path.isdir(PCA_model_save_dir) == False:
        os.makedirs(PCA_model_save_dir)
    joblib.dump(pca, os.path.join(PCA_model_save_dir, 'pca_subj_' + format(sub, '02') +'.pkl'))

    # Saving the downsampled features:
    save_dir = os.path.join(output_dir, 'dnn_feature_maps', 'subj' + format(sub, '02'), 'train_data')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    print("Saving downsampled features...")
    i = 0 # Initialize counter
    for batch_file in tqdm(fmaps_train):
        batch_data = np.load(batch_file) # Load batch data
        batch_data = sc.transform(batch_data) # Standardize batch using the scaler computer previously
        batch_data_pca = pca.transform(batch_data) # Apply PCA to reduce dimensionality to 100 components
        np.save(os.path.join(save_dir, f'train_data_batch_features_pca_{i}'), batch_data_pca) # Save data
        i += 1
    del fmaps_train # Free computing resources


    # =============================================================================
    # Apply PCA on the test images feature maps
    # =============================================================================
    print("Standardizing and applying PCA on the test image features maps...")
    save_dir = os.path.join(output_dir, 'dnn_feature_maps', 'subj' + format(sub, '02'), 'test_data')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    # Load previously computed standardizer and PCA model
    sc = joblib.load(os.path.join(Scaler_model_save_dir, 'scaler_subj_' + format(sub, '02') + '.pkl'))
    pca = joblib.load(os.path.join(PCA_model_save_dir, 'pca_subj_' + format(sub, '02') +'.pkl'))

    i = 0
    for batch_file in tqdm(fmaps_test):
        batch_data = np.load(batch_file)
        batch_data = sc.transform(batch_data)  # Standardize Batch
        batch_data_pca = pca.transform(batch_data)  # Apply PCA
        np.save(os.path.join(save_dir, f'test_data_batch_features_pca_{i}'), batch_data_pca)
        i += 1
    del fmaps_test  # Free computing resources

# =============================================================================
# Algorithm to train the linear regressor on both brain hemisphere
# =============================================================================

def baseline_encoding_train_encoding_model(sub, project_dir, output_dir):
    print('>>> Algonauts 2023 train encoding <<<')
    print('\nInput arguments:')
    print(f'\n Using subject: {sub}')
    print(f'\n Project directory: {project_dir}')

    # =============================================================================
    # Load the DNN feature maps
    # =============================================================================
    baseline_encoding_dir = os.path.join(output_dir, 'baseline_encoding_model')

    # Load training PCA reduced maps
    dnn_dir = os.path.join(baseline_encoding_dir, 'dnn_feature_maps','subj' + format(sub, '02'))
    train_batch_files = sorted(glob.glob(os.path.join(dnn_dir, 'train_data', 'train_data_batch_features_pca_*.npy')))

    # Concatenate all PCA-reduced batches
    X_train_batches = [np.load(f) for f in train_batch_files]
    X_train = np.concatenate(X_train_batches, axis=0)

    # Load test PCA reduced maps
    test_batch_files = sorted(glob.glob(os.path.join(dnn_dir, 'test_data', 'test_data_batch_features_pca_*.npy')))
    X_test_batches = [np.load(f) for f in test_batch_files]
    X_test = np.concatenate(X_test_batches, axis=0)

    # =============================================================================
    # Load the fMRI data
    # =============================================================================

    data_dir = os.path.join(project_dir, 'Dataset', 'train_data',
                            'subj' + format(sub, '02'),
                            'training_split', 'training_fmri')
    y_train_lh = np.load(os.path.join(data_dir, 'lh_training_fmri.npy'))
    y_train_rh = np.load(os.path.join(data_dir, 'rh_training_fmri.npy'))

    # =============================================================================
    # Train the linear regression and save the predicted fMRI for the test images
    # =============================================================================

    # Create empty synthetic fMRI data matrices of dimension
    # (Test image conditions × fMRI vertices)

    synt_test_lh = np.zeros((X_test.shape[0], y_train_lh.shape[1]))
    synt_test_rh = np.zeros((X_test.shape[0], y_train_rh.shape[1]))

    # Independently for each fMRI vertex, fit a linear regression using the
    # training image conditions and use it to synthesize the fMRI responses of the
    # test image conditions
    print("\n Training linear regressor for the left hemisphere")
    for v in tqdm(range(y_train_lh.shape[1])):
        # Fit linear regression model using the training image conditions (X_train) and the
        # fMRI responses for vertex v in the LEFT hemisphere
        reg_lh = LinearRegression().fit(X_train, y_train_lh[:, v])
        # Uses the fitted model to predict fMRI responses for the test image conditions
        # (X_test) and store the prediction in the corresponding column
        synt_test_lh[:, v] = reg_lh.predict(X_test)
    print("Training linear regressor for the right hemisphere")
    for v in tqdm(range(y_train_rh.shape[1])):
        # Fit linear regression model using the training image conditions (X_train) and the
        # fMRI responses for vertex v in the RIGHT hemisphere
        reg_rh = LinearRegression().fit(X_train, y_train_rh[:, v])
        # Uses the fitted model to predict fMRI responses for the test image conditions
        # (X_test) and store the prediction in the corresponding column
        synt_test_rh[:, v] = reg_rh.predict(X_test)

    # Save the synthetic fMRI test data
    save_dir = os.path.join(baseline_encoding_dir, 'synthetic_data', 'subj' + format(sub, '02'))
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'lh_test_synthetic_fmri.npy'), synt_test_lh)
    np.save(os.path.join(save_dir, 'rh_test_synthetic_fmri.npy'), synt_test_rh)

# =============================================================================
# Algorithm to test the linear regressor in correlation to test data
# =============================================================================

def baseline_encoding_test_model(sub, project_dir, output_dir):
    print('>>> Algonauts 2023 test encoding <<<')
    print(f'Subject used: {sub}')
    print(f'Project directory: {project_dir}')

    # =============================================================================
    # Load the biological fMRI test data
    # =============================================================================
    data_dir = os.path.join(project_dir, 'Dataset', 'test_data', 'subj' +
                            format(sub, '02'), 'test_split', 'test_fmri')
    lh_bio_test = np.load(os.path.join(data_dir, 'lh_test_fmri.npy'))
    rh_bio_test = np.load(os.path.join(data_dir, 'rh_test_fmri.npy'))

    # =============================================================================
    # Load the synthetic fMRI test data
    # =============================================================================

    data_dir = os.path.join(output_dir, 'synthetic_data', 'subj' +format(sub, '02'))
    lh_synt_test = np.load(os.path.join(data_dir, 'lh_test_synthetic_fmri.npy'))
    rh_synt_test = np.load(os.path.join(data_dir, 'rh_test_synthetic_fmri.npy'))

    # =============================================================================
    # Correlate the biological and synthetic fMRI test data
    # =============================================================================
    # Left hemisphere
    tqdm.write("\nCalculating the pearson correlation coefficient between LH synthetic data and LH biological data")
    lh_correlation = np.zeros(lh_bio_test.shape[1])
    for v in tqdm(range(lh_bio_test.shape[1])):
        lh_correlation[v] = pearsonr(lh_bio_test[:, v], lh_synt_test[:, v])[0]

    tqdm.write("\nCalculating the pearson correlation coefficient between RH synthetic data and RH biological data")
    # Right hemishpere
    rh_correlation = np.zeros(rh_bio_test.shape[1])
    for v in tqdm(range(rh_bio_test.shape[1])):
        rh_correlation[v] = pearsonr(rh_bio_test[:, v], rh_synt_test[:, v])[0]

    # =============================================================================
    # Load the noise ceiling
    # =============================================================================
    data_dir = os.path.join(project_dir, 'Dataset', 'test_data', 'subj' +
                            format(sub, '02'), 'test_split', 'noise_ceiling')
    lh_noise_ceiling = np.load(os.path.join(data_dir, 'lh_noise_ceiling.npy'))
    rh_noise_ceiling = np.load(os.path.join(data_dir, 'rh_noise_ceiling.npy'))

    # =============================================================================
    # Compute the noise-ceiling-normalized encoding accuracy
    # =============================================================================
    # Set negative correlation values to 0, so to keep the noise-normalized
    # encoding accuracy positive
    lh_correlation[lh_correlation < 0] = 0
    rh_correlation[rh_correlation < 0] = 0

    # Square the correlation values into r2 scores
    lh_r2 = lh_correlation ** 2
    rh_r2 = rh_correlation ** 2

    # Add a very small number to noise ceiling values of 0, otherwise the noise-
    # normalized encoding accuracy cannot be calculated (division by 0 is not
    # possible)
    lh_idx_nc_zero = np.argwhere(lh_noise_ceiling == 0)
    rh_idx_nc_zero = np.argwhere(rh_noise_ceiling == 0)
    lh_noise_ceiling[lh_idx_nc_zero] = 1e-14
    rh_noise_ceiling[rh_idx_nc_zero] = 1e-14

    # Compute the noise-ceiling-normalized encoding accuracy
    lh_noise_normalized_encoding = np.divide(lh_r2, lh_noise_ceiling) * 100
    rh_noise_normalized_encoding = np.divide(rh_r2, rh_noise_ceiling) * 100

    # Set the noise-normalized encoding accuracy to 100 for those vertices in which
    # the correlation is higher than the noise ceiling, to prevent encoding
    # accuracy values higher than 100%
    lh_noise_normalized_encoding[lh_noise_normalized_encoding > 100] = 100
    rh_noise_normalized_encoding[rh_noise_normalized_encoding > 100] = 100

    # =============================================================================
    # Save the noise-ceiling-normalized encoding accuracy
    # =============================================================================
    encoding_accuracy = {
        'lh_r2': lh_r2,
        'rh_r2': rh_r2,
        'lh_noise_ceiling': lh_noise_ceiling,
        'rh_noise_ceiling': rh_noise_ceiling,
        'lh_noise_normalized_encoding': lh_noise_normalized_encoding,
        'rh_noise_normalized_encoding': rh_noise_normalized_encoding,
        'lh_idx_nc_zero': lh_idx_nc_zero,
        'rh_idx_nc_zero': rh_idx_nc_zero
    }

    save_dir = os.path.join(output_dir, 'encoding_accuracy')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    file_name = 'encoding_accuracy_subj' + format(sub, '02')

    np.save(os.path.join(save_dir, file_name), encoding_accuracy)

# ======================================================================================================
# Algorithm to plot the encoding models noise-ceiling-normalized encoding accuracy on a brain surface
# WARNING: This algorithm DOES NOT WORK ON WINDOWS because it uses packages only available on Linux/MacOS
# please use a UNIX machine to run it
# ======================================================================================================

def baseline_encoding_plot_results(subjects, project_dir, output_dir):
    print('>>> Algonauts 2023 plot encoding accuracy for all subjects <<<')
    print(f"Project diretory: {project_dir}")

    # =============================================================================
    # Load the noise-ceiling-normalized encoding accuracy for all NSD subjects
    # =============================================================================
    lh_scores = []
    rh_scores = []

    for s in subjects:
        data_dir = os.path.join(output_dir, 'encoding_accuracy',
                                'encoding_accuracy_subj' + format(s, '02') + '.npy')
        data = np.load(data_dir, allow_pickle=True).item()
        lh_scores.append(data['lh_noise_normalized_encoding'])
        rh_scores.append(data['rh_noise_normalized_encoding'])

    # =============================================================================
    # Map the data to fsaverage space
    # =============================================================================
    lh_fsaverage = []
    rh_fsaverage = []
    for s, sub in enumerate(subjects):
        # Left hemisphere
        lh_mask_dir = os.path.join(project_dir, 'Dataset', 'train_data', 'subj' +
                                   format(sub, '02'), 'roi_masks', 'lh.all-vertices_fsaverage_space.npy')
        lh_fsaverage_nsd_general_plus = np.load(lh_mask_dir)
        lh_fsavg = np.empty((len(lh_fsaverage_nsd_general_plus)))
        lh_fsavg[:] = np.nan
        lh_fsavg[np.where(lh_fsaverage_nsd_general_plus)[0]] = lh_scores[s]
        lh_fsaverage.append(copy(lh_fsavg))
        # Right hemisphere
        rh_mask_dir = os.path.join(project_dir, 'Dataset', 'train_data', 'subj' +
                                   format(sub, '02'), 'roi_masks', 'rh.all-vertices_fsaverage_space.npy')
        rh_fsaverage_nsd_general_plus = np.load(rh_mask_dir)
        rh_fsavg = np.empty((len(rh_fsaverage_nsd_general_plus)))
        rh_fsavg[:] = np.nan
        rh_fsavg[np.where(rh_fsaverage_nsd_general_plus)[0]] = rh_scores[s]
        rh_fsaverage.append(copy(rh_fsavg))

    # Average the scores across subjects
    lh_fsaverage = np.nanmean(lh_fsaverage, 0)
    rh_fsaverage = np.nanmean(rh_fsaverage, 0)

    # =============================================================================
    # Plot parameters for colorbar
    # =============================================================================
    plt.rc('xtick', labelsize=19)
    plt.rc('ytick', labelsize=19)

    # =============================================================================
    # Plot the results on brain surfaces
    # =============================================================================
    subject = 'fsaverage'
    data = np.append(lh_fsaverage, rh_fsaverage)
    vertex_data = cortex.Vertex(data, subject, cmap='hot', vmin=0, vmax=100)
    cortex.quickshow(vertex_data)
    plt.show()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    # plt.savefig('algonauts_2023_challenge_winner_1.png', transparent=True, dpi=100)

    # =============================================================================
    # Plot the fsaverage surface templates
    # =============================================================================
    # Plot the full surface
    data = np.append(lh_fsaverage, rh_fsaverage)
    data[:] = np.nan  # 40
    vertex_data = cortex.Vertex(data, subject, cmap='Greys', vmin=0, vmax=100,
                                with_colorbar=False)
    cortex.quickshow(vertex_data, with_curvature=True, with_colorbar=False)
    plt.show()
    # plt.savefig('algonauts_2023_full_surface_template.png', transparent=True, dpi=100)

    # Plot the challenge vertices surface --> ['PiYG', 'RdPu_r']
    data = np.append(lh_fsaverage, rh_fsaverage)
    idx = ~np.isnan(data)
    data[idx] = 5
    vertex_data = cortex.Vertex(data, subject, cmap='PiYG', vmin=0, vmax=100)
    cortex.quickshow(vertex_data, with_colorbar=False)
    plt.show()
    # plt.savefig('algonauts_2023_challenge_vertices_surface.png', transparent=True, dpi=100)