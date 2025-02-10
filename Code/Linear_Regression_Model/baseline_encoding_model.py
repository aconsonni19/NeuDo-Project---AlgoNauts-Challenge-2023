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
import sys
from copy import copy

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
    """
    Extracts the features from the Dataset Images
    :param sub: The subject used
    :param project_dir: The project directory where the Dataset is contained
    :param output_dir: The output directory
    :return:
    """
    # =============================================================================
    # Setup of the enviorment
    # =============================================================================
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

    # Define the image preprocessing
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
    """
    Train the linear regression model using the features extracted from the dataset
    :param sub: The subject used
    :param project_dir: The project directory where the Dataset is contained
    :param output_dir: The project Output directory, where the Outputs of the model are contained
    :return:
    """


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
    """
    Test the linear regression model using the test split of the dataset
    :param sub: The subject used; determines which linear regressor will be used
    :param project_dir: The project directory where the Dataset is contained
    :param output_dir: The project Output directory, where the Outputs of the model are contained
    :return:
    """
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

    # Returns the pearson correlation in a dictionary in order to use them to plot results
    # NOTE: Maybe save this also as a numpy file
    model_correlations = {
        'lh_pearson_correlations': lh_correlation,
        'rh_pearson_correlations': rh_correlation,
    }

    return model_correlations

# ======================================================================================================
# Algorithm to plot the encoding model R^2 accuracy score for each emisphere in challenge space
# ======================================================================================================

def baseline_encoding_accuracy_plot(sub, project_dir, output_dir):
    """
    Plots the mean encoding accuracy for both hemispheres in a barplot
    :param sub: The subject for which the results will be plotted
    :param project_dir: The project directory
    :param output_dir: The project outputs directory
    :return:
    """
    print(f'>>> Algonauts 2023 Bar Plot of Encoding Accuracy (R^2 Scores) for Subject {sub} <<<')
    print(f"Project directory: {project_dir}")

    # Load the noise-normalized encoding accuracy (R^2 scores) for the given subject
    data_dir = os.path.join(output_dir, 'encoding_accuracy',
                            f'encoding_accuracy_subj{format(sub, "02")}.npy')

    if not os.path.exists(data_dir):
        print(f"Error: Encoding accuracy file not found for Subject {sub}")
        return

    data = np.load(data_dir, allow_pickle=True).item()

    # Compute mean R?^2 score for left and right hemispheres
    lh_mean = np.nanmean(data['lh_noise_normalized_encoding'])  # Left hemisphere mean R^2
    rh_mean = np.nanmean(data['rh_noise_normalized_encoding'])  # Right hemisphere mean R^2

    # Compute standard deviation (for variability visualization)
    lh_std = np.nanstd(data['lh_noise_normalized_encoding'])
    rh_std = np.nanstd(data['rh_noise_normalized_encoding'])

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x_labels = ['Left Hemisphere', 'Right Hemisphere']
    x_pos = np.arange(len(x_labels))
    bar_width = 0.5

    ax.bar(x_pos, [lh_mean, rh_mean], yerr=[lh_std, rh_std], width=bar_width,
           color=['blue', 'red'], alpha=0.7, capsize=5, label="Mean R² Score")

    # Labels and title
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Mean Noise-normalized $R^2$")
    ax.set_title(f"Encoding Accuracy (Noise-normalized $R^2$) for Subject {sub}")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Save the plot
    save_dir = os.path.join(output_dir, 'accuracy_barplot')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    plot_path = os.path.join(save_dir, f'encoding_accuracy_subj{format(sub, "02")}_barplot.png')
    plt.savefig(plot_path)
    print(f"Plot saved at: {plot_path}")

# ======================================================================================================
# Algorithm to plot the encoding model R^2 accuracy score for each ROI in fsavarage space
# ======================================================================================================

def baseline_encoding_accuracy_plot_fsavarage(sub, project_dir, output_dir):
    """
    Plots the encoding accuracy for each ROI in fsavarage space
    :param sub: The subject for which the results will be plotted
    :param project_dir: The project directory
    :param output_dir: The project outputs directory
    :return:
    """

    print(f'>>> Encoding Accuracy Barplot (fsaverage) for Subject {sub} <<<')

    # Load the noise-ceiling-normalized encoding accuracy
    data_dir = os.path.join(output_dir, 'encoding_accuracy', f'encoding_accuracy_subj{sub:02}.npy')
    data = np.load(data_dir, allow_pickle=True).item()
    lh_scores = data['lh_noise_normalized_encoding']
    rh_scores = data['rh_noise_normalized_encoding']

    # Map the data to fsaverage space
    lh_mask_dir = os.path.join(project_dir, 'Dataset', 'train_data', f'subj{sub:02}', 'roi_masks',
                               'lh.all-vertices_fsaverage_space.npy')
    rh_mask_dir = os.path.join(project_dir, 'Dataset', 'train_data', f'subj{sub:02}', 'roi_masks',
                               'rh.all-vertices_fsaverage_space.npy')

    lh_fsaverage_nsd_general_plus = np.load(lh_mask_dir)
    rh_fsaverage_nsd_general_plus = np.load(rh_mask_dir)

    lh_fsavg = np.empty(len(lh_fsaverage_nsd_general_plus))
    rh_fsavg = np.empty(len(rh_fsaverage_nsd_general_plus))
    lh_fsavg[:] = np.nan
    rh_fsavg[:] = np.nan

    lh_fsavg[np.where(lh_fsaverage_nsd_general_plus)[0]] = lh_scores
    rh_fsavg[np.where(rh_fsaverage_nsd_general_plus)[0]] = rh_scores

    # Load the ROI mappings
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
                         'mapping_floc-faces.npy', 'mapping_floc-places.npy',
                         'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(project_dir, 'Dataset', 'train_data', f'subj{sub:02}',
                                                  'roi_masks', r), allow_pickle=True).item())

    # Load the ROI masks in fsaverage space
    lh_challenge_roi_files = ['lh.prf-visualrois_fsaverage_space.npy',
                              'lh.floc-bodies_fsaverage_space.npy', 'lh.floc-faces_fsaverage_space.npy',
                              'lh.floc-places_fsaverage_space.npy', 'lh.floc-words_fsaverage_space.npy',
                              'lh.streams_fsaverage_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_fsaverage_space.npy',
                              'rh.floc-bodies_fsaverage_space.npy', 'rh.floc-faces_fsaverage_space.npy',
                              'rh.floc-places_fsaverage_space.npy', 'rh.floc-words_fsaverage_space.npy',
                              'rh.streams_fsaverage_space.npy']

    lh_challenge_rois = [np.load(os.path.join(project_dir, 'Dataset', 'train_data', f'subj{sub:02}',
                                              'roi_masks', f)) for f in lh_challenge_roi_files]
    rh_challenge_rois = [np.load(os.path.join(project_dir, 'Dataset', 'train_data', f'subj{sub:02}',
                                              'roi_masks', f)) for f in rh_challenge_roi_files]

    # Compute mean R^2 scores for each ROI
    roi_names = []
    lh_roi_scores = []
    rh_roi_scores = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0:
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_scores.append(lh_fsavg[lh_roi_idx])
                rh_roi_scores.append(rh_fsavg[rh_roi_idx])

    # Add an "All vertices" category
    roi_names.append('All vertices')
    lh_roi_scores.append(lh_fsavg)
    rh_roi_scores.append(rh_fsavg)

    # Compute mean R^2 scores for plotting
    lh_mean_roi_scores = [np.nanmean(lh_roi_scores[r]) for r in range(len(lh_roi_scores))]
    rh_mean_roi_scores = [np.nanmean(rh_roi_scores[r]) for r in range(len(rh_roi_scores))]

    # Create the bar plot
    plt.figure(figsize=(18, 10))
    plt.title(f'Noise-Ceiling-Normalized R² Scores for Subject {sub} (fsaverage)')
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width / 2, lh_mean_roi_scores, width, label='Left Hemisphere')
    plt.bar(x + width / 2, rh_mean_roi_scores, width, label='Right Hemisphere')
    plt.xlim(left=min(x) - .5, right=max(x) + .5)
    plt.ylim(bottom=0, top=100)
    plt.xlabel('ROIs')
    plt.xticks(ticks=x, labels=roi_names, rotation=60)
    plt.ylabel('Mean $R^2$ Score (%)')
    plt.legend(frameon=True, loc=1)

    # Save the plot
    save_dir = os.path.join(output_dir, 'fsaverage_accuracy_barplot')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'subj{sub:02}_fsaverage_r2_barplot.png'))

# =========================================================================================================
# Algorithm that creates a barplot of the pearson correlation of the model for both hemispheres in each ROI
# =========================================================================================================

def baseline_encoding_plot_roi_correlation(sub, project_dir, output_dir, model_correlations):
    """
    Plots the pearson correlation of the model results for each ROI for both hemispheres
    :param sub: The subject for which the results will be plotted
    :param project_dir: The project directory
    :param output_dir: The project outputs directory
    :param model_correlations: A dictionary with the model pearson correlation index for both hemispheres
    :return:
    """

    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
                         'mapping_floc-faces.npy', 'mapping_floc-places.npy',
                         'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(project_dir, 'Dataset', 'train_data', 'subj' + format(sub, '02'),
                                                  'roi_masks', r),allow_pickle=True).item())
    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
                              'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
                              'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
                              'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
                              'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
                              'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
                              'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(project_dir, 'Dataset', 'train_data', 'subj' + format(sub, '02'),
                                                      'roi_masks',lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(project_dir, 'Dataset', 'train_data', 'subj' + format(sub, '02'),
                                                      'roi_masks', rh_challenge_roi_files[r])))

    lh_correlations = model_correlations['lh_pearson_correlations']
    rh_correlations = model_correlations['rh_pearson_correlations']

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0:  # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlations[lh_roi_idx])
                rh_roi_correlation.append(rh_correlations[rh_roi_idx])
    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlations)
    rh_roi_correlation.append(rh_correlations)

    # Create the plot
    lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r])
                               for r in range(len(lh_roi_correlation))]
    rh_mean_roi_correlation = [np.mean(rh_roi_correlation[r])
                               for r in range(len(rh_roi_correlation))]
    plt.figure(figsize=(18, 10))
    plt.title(f'Model pearson correlation on each ROI for subject {sub}' )
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width / 2, lh_mean_roi_correlation, width, label='Left Hemisphere')
    plt.bar(x + width / 2, rh_mean_roi_correlation, width,
            label='Right Hemishpere')
    plt.xlim(left=min(x) - .5, right=max(x) + .5)
    plt.ylim(bottom=0, top=1)
    plt.xlabel('ROIs')
    plt.xticks(ticks=x, labels=roi_names, rotation=60)
    plt.ylabel('Mean Pearson\'s $r$')
    plt.legend(frameon=True, loc=1)
    save_dir = os.path.join(output_dir,'baseline_encoding_model', 'correlation_barplot')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'subj' + format(sub, '02') + '_model_pearson_correlation.png'))




