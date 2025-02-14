import gc
import glob
import os.path
import random

import numpy as np
import clip
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import MultiLayerPerceptron as mlp
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable as V
from tqdm import tqdm
import torch
from torchvision import transforms as trn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Algorithm to extract features using CLIP
# =============================================================================
def extract_clip_features(sub, clip_model_name, project_dir, output_dir):
    """
    Extracts features from Dataset Images using CLIP.
    :param clip_model_name:
    :param sub: The subject used
    :param project_dir: The project directory where the dataset is stored
    :param output_dir: The output directory for extracted features
    """
    print('>>> Extracting image features using CLIP <<<')
    print(f'Project directory: {project_dir}')
    print(f'Output directory: {output_dir}')
    print(f'Subject used: {sub}')

    # Set up environment
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.empty_cache()
    gc.collect()

    # Set random seed for reproducibility
    seed = 20200220
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    # Define the image preprocessing

    centre_crop = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip.load(clip_model_name, device=device)[0]
    clip_model.eval()

    # Image directories
    img_set_dir = os.path.join(project_dir, 'Dataset')
    splits_parent = ['train_data', 'test_data']
    splits = ['training_split', 'test_split']
    splits_child = ['training_images', 'test_images']
    batch_size = 1024
    fmaps_train = []
    fmaps_test = []
    batch_files = [fmaps_train, fmaps_test]

    for s in range(len(splits_parent)):
        image_list = sorted(os.listdir(os.path.join(img_set_dir, splits_parent[s],
                                                    f'subj{sub:02}', splits[s], splits_child[s])))

        print(f"Extracting features from {splits_parent[s]} in batches of {batch_size}")

        for i in range(0, len(image_list), batch_size):
            batch_images = image_list[i:i + batch_size]
            batch_features = []

            # Process each image in the batch
            for image in tqdm(batch_images, desc=f"Batch {i // batch_size + 1}"):
                img_path = os.path.join(img_set_dir, splits_parent[s], f'subj{sub:02}', splits[s], splits_child[s],
                                        image)
                img = Image.open(img_path).convert('RGB')

                # Preprocess and move to device
                input_img = V(centre_crop(img).unsqueeze(0)).to(
                    device)  # Crop image to center and adds dimension for batch
                # processing

                # Extract CLIP features
                with torch.no_grad():
                    img_feats = clip_model.encode_image(input_img).cpu().numpy().flatten()

                batch_features.append(img_feats)

            # Convert batch features to NumPy array
            batch_features = np.asarray(batch_features)

            # Save batch features
            save_dir = os.path.join(output_dir, 'clip_features', splits_parent[s], f'subj{sub:02}')
            os.makedirs(save_dir, exist_ok=True)
            batch_file = os.path.join(save_dir, f'{splits_parent[s]}_batch_features_{i // batch_size}.npy')
            np.save(batch_file, batch_features)
            batch_files[s].append(batch_file)
    print(">>> Feature extraction complete <<<")


# =============================================================================
# Algorithm to train the MLP using the CLIP image embeddings
# =============================================================================
def train_mlp_models(sub, project_dir, output_dir, trials = 5):
    """
    Train the MLP model using features extracted from the dataset.
    """
    print(f">>> Training MLP encoding models for subject {sub} <<<")
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =============================================================================
    # Load CLIP embeddings
    # =============================================================================
    clip_features_dir = os.path.join(output_dir, 'clip_features')

    train_embeddings_dir = os.path.join(clip_features_dir, 'train_data', f'subj{sub:02}')
    test_embeddings_dir = os.path.join(clip_features_dir, 'test_data', f'subj{sub:02}')

    # Load training features
    train_batch_files = sorted(glob.glob(os.path.join(train_embeddings_dir, 'train_data_batch_features_*.npy')))
    X_train = np.concatenate([np.load(f) for f in train_batch_files], axis=0)

    # Load test features
    test_batch_files = sorted(glob.glob(os.path.join(test_embeddings_dir, 'test_data_batch_features_*.npy')))
    X_test = np.concatenate([np.load(f) for f in test_batch_files], axis=0)

    # =============================================================================
    # Load fMRI data
    # =============================================================================
    train_fmri_dir = os.path.join(project_dir, 'Dataset', 'train_data', f'subj{sub:02}', 'training_split', 'training_fmri')
    y_train_lh = np.load(os.path.join(train_fmri_dir, 'lh_training_fmri.npy'))
    y_train_rh = np.load(os.path.join(train_fmri_dir, 'rh_training_fmri.npy'))
    y_train = np.concatenate([y_train_lh, y_train_rh], axis=1)  # Combine hemispheres

    # Load test fMRI data
    test_fmri_dir = os.path.join(project_dir, 'Dataset', 'test_data', f'subj{sub:02}', 'test_split', 'test_fmri')
    y_test_lh = np.load(os.path.join(test_fmri_dir, 'lh_test_fmri.npy'))
    y_test_rh = np.load(os.path.join(test_fmri_dir, 'rh_test_fmri.npy'))
    y_test = np.concatenate([y_test_lh, y_test_rh], axis=1)  # Combine hemispheres

    # Normalize input features
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)  # Apply same transformation to test set


    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Define model, optimizer, and loss function
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]  # Number of fMRI vertices

    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    hyperparam_space = {
        'lr': [0.1, 0.01, 0.001, 0.0001],
        'weight_decay': [0.1, 0.01, 0.001, 0.0001],


    }

    for trial in range(trials):

        lr = random.choice(hyperparam_space['lr'])
        weight_decay = random.choice(hyperparam_space['weight_decay'])
        print(f"Trial: {trial} \n lr: {lr} \n wd: {weight_decay}")
        for model in mlp.get_all_available_mlp(input_dim, output_dim):
            print(f'Training model: {model.name}')
            model = model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                                   verbose=True)
            criterion = nn.SmoothL1Loss()

            # =============================================================================
            # Train the MLP
            # =============================================================================
            epochs = 50
            loss_history = []
            test_loss_history = []
            torch.cuda.empty_cache()
            patience = 3
            best_loss = float("inf")

            print("\nTraining MLP encoding model...")
            for epoch in range(epochs):
                total_loss = 0
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():  # Enables mixed precision to save memory
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                loss_history.append(avg_train_loss)

                # Validation step
                model.eval()
                total_test_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        y_test_pred = model(batch_X)
                        loss = criterion(y_test_pred, batch_y)
                        total_test_loss += loss.item()

                avg_test_loss = total_test_loss / len(test_loader)
                test_loss_history.append(avg_test_loss)

                scheduler.step(avg_train_loss)
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

                # Early stopping check
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    counter = 0  # Reset counter
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            torch.cuda.empty_cache()
            print(">>> Training Complete! Synthetic fMRI data saved. <<<")

            # =============================================================================
            # Save synthetic fMRI test data
            # =============================================================================
            print("\nSaving synthetic fMRI data...")

            y_test_pred_numpy = y_test_pred.cpu().numpy()
            num_lh = y_train_lh.shape[1]
            synt_test_lh = y_test_pred_numpy[:, :num_lh]
            synt_test_rh = y_test_pred_numpy[:, num_lh:]

            save_dir = os.path.join(output_dir, model.name, 'synthetic_data', f'subj{sub:02}')
            os.makedirs(save_dir, exist_ok=True)

            np.save(os.path.join(save_dir, f'lh_test_synthetic_fmri_trial_{trial}.npy'), synt_test_lh)
            np.save(os.path.join(save_dir, f'rh_test_synthetic_fmri_trial_{trial}.npy'), synt_test_rh)

            # =============================================================================
            # Plot Loss Curve
            # =============================================================================
            plt.figure(figsize=(15, 6))
            plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b',
                     label='Train Loss')
            plt.plot(range(1, len(test_loss_history) + 1), test_loss_history, marker='s', linestyle='-', color='r',
                     label='Test Loss')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training & Test Loss Curve of model {model.name} with lr = {lr} and wd = {weight_decay}')
            plt.legend()
            plt.grid(True)

            # Save the plot
            loss_plot_dir = os.path.join(output_dir, model.name, 'loss_graph')
            os.makedirs(loss_plot_dir, exist_ok=True)
            del model, optimizer, scheduler, criterion
            torch.cuda.empty_cache()  # Free GPU memory
            gc.collect()  # Force garbage collection

            plot_path = os.path.join(loss_plot_dir, f'loss_graph_subj{format(sub, "02")}_plot_trial_{trial}.png')
            plt.savefig(plot_path)
            print(f"Plot saved at: {plot_path}")


# =============================================================================
# Algorithm to test the MLP on the test data
# =============================================================================
def test_mlp_encoding(sub, project_dir, output_dirs):
    """
    Test the MLP encoding model using the test split of the dataset.
    Computes the Pearson correlation between biological and synthetic fMRI data.

    :param sub: The subject used; determines which MLP model's predictions are tested.
    :param project_dir: The project directory where the dataset is contained.
    :param output_dirs: The directory where model outputs are stored.
    :return: Dictionary containing Pearson correlations for both hemispheres.
    """

    print('>>> Testing MLP encoding model <<<')
    print(f'Subject: {sub}')
    print(f'Project directory: {project_dir}')

    # =============================================================================
    # Load the Biological fMRI Test Data
    # =============================================================================
    data_dir = os.path.join(project_dir, 'Dataset', 'test_data', f'subj{sub:02}', 'test_split', 'test_fmri')
    lh_bio_test = np.load(os.path.join(data_dir, 'lh_test_fmri.npy'))  # Left Hemisphere
    rh_bio_test = np.load(os.path.join(data_dir, 'rh_test_fmri.npy'))  # Right Hemisphere

    # =============================================================================
    # Load the MLP-Synthesized fMRI Test Data
    # =============================================================================
    data_dir = os.path.join(output_dirs, 'synthetic_data', f'subj{sub:02}')
    lh_synthetic_files = sorted(glob.glob(os.path.join(data_dir, 'lh_test_synthetic_fmri_trial_*.npy')))
    rh_synthetic_files = sorted(glob.glob(os.path.join(data_dir, 'rh_test_synthetic_fmri_trial_*.npy')))


    # =============================================================================
    # Compute Pearson Correlation Between Biological and Synthetic fMRI Data
    # =============================================================================
    trial = 0
    lh_correlations = []
    rh_correlations = []
    for lh_synt_test_file, rh_synt_test_file in zip(lh_synthetic_files, rh_synthetic_files):
        lh_synt_test = np.load(lh_synt_test_file)
        print(f"\nComputing Pearson correlation for LH for trial {trial}...")
        lh_correlation = np.zeros(lh_bio_test.shape[1])
        if not np.any(np.isnan(lh_synt_test)) or np.any(np.isinf(lh_synt_test)):
            for v in tqdm(range(lh_bio_test.shape[1])):
                lh_correlation[v] = pearsonr(lh_bio_test[:, v], lh_synt_test[:, v])[0]
        rh_synt_test = np.load(rh_synt_test_file)
        rh_correlation = np.zeros(rh_bio_test.shape[1])
        print(f"\nComputing Pearson correlation for RH for trial {trial}...")
        if not np.any(np.isnan(rh_synt_test)) or np.any(np.isinf(rh_synt_test)):
            for v in tqdm(range(rh_bio_test.shape[1])):
                rh_correlation[v] = pearsonr(rh_bio_test[:, v], rh_synt_test[:, v])[0]

        # =============================================================================
        # Load the Noise Ceiling Data
        # =============================================================================
        data_dir = os.path.join(project_dir, 'Dataset', 'test_data', f'subj{sub:02}', 'test_split', 'noise_ceiling')
        lh_noise_ceiling = np.load(os.path.join(data_dir, 'lh_noise_ceiling.npy'))
        rh_noise_ceiling = np.load(os.path.join(data_dir, 'rh_noise_ceiling.npy'))

        # =============================================================================
        # Compute Noise-Ceiling-Normalized Encoding Accuracy
        # =============================================================================
        lh_correlation[lh_correlation < 0] = 0  # Remove negative correlations
        rh_correlation[rh_correlation < 0] = 0

        lh_correlations.append(lh_correlation)
        rh_correlations.append(rh_correlation)

        lh_r2 = lh_correlation ** 2  # Squaring to get R² scores
        rh_r2 = rh_correlation ** 2

        # Handle cases where the noise ceiling is zero (avoid division by zero)
        lh_noise_ceiling[lh_noise_ceiling == 0] = 1e-14
        rh_noise_ceiling[rh_noise_ceiling == 0] = 1e-14

        lh_noise_normalized_encoding = (lh_r2 / lh_noise_ceiling) * 100
        rh_noise_normalized_encoding = (rh_r2 / rh_noise_ceiling) * 100

        # Cap accuracy at 100% (to prevent overestimation)
        lh_noise_normalized_encoding[lh_noise_normalized_encoding > 100] = 100
        rh_noise_normalized_encoding[rh_noise_normalized_encoding > 100] = 100

        # =============================================================================
        # Save the Results
        # =============================================================================
        encoding_accuracy = {
            'lh_r2': lh_r2,
            'rh_r2': rh_r2,
            'lh_noise_ceiling': lh_noise_ceiling,
            'rh_noise_ceiling': rh_noise_ceiling,
            'lh_noise_normalized_encoding': lh_noise_normalized_encoding,
            'rh_noise_normalized_encoding': rh_noise_normalized_encoding
        }

        save_dir = os.path.join(output_dirs, 'encoding_accuracy')
        os.makedirs(save_dir, exist_ok=True)

        file_name = f'encoding_accuracy_subj{sub:02}_trial_{trial}.npy'
        np.save(os.path.join(save_dir, file_name), encoding_accuracy)
        trial += 1

    # =============================================================================
    # Return Correlation Results for Visualization
    # =============================================================================
    model_correlations = {
        'lh_pearson_correlations': lh_correlations,
        'rh_pearson_correlations': rh_correlations
    }

    return model_correlations


# =============================================================================
# Algorithm to plot the accuracy of the MLP model
# =============================================================================
def mlp_accuarcy_plot(sub, project_dir, output_dir):
    """
    Plots the mean encoding accuracy for both hemispheres in a barplot
    :param sub: The subject for which the results will be plotted
    :param project_dir: The project directory
    :param output_dir: The project outputs directory
    :return:
    """
    print(f'>>> MLP Plot accuracy <<<')
    print(f"Project directory: {project_dir}")

    # Load the noise-normalized encoding accuracy (R^2 scores) for the given subject
    data_dir = os.path.join(output_dir, 'encoding_accuracy')

    accuracy_files = glob.glob(os.path.join(data_dir, f'encoding_accuracy_subj{format(sub, "02")}_trial_*.npy'))

    trial = 0
    for accuracy_file in accuracy_files:
        if not os.path.exists(accuracy_file):
            print(f"Error: Encoding accuracy file not found for Subject {sub}")
            return

        data = np.load(accuracy_file, allow_pickle=True).item()

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
        ax.set_title(f"Encoding Accuracy (Noise-normalized $R^2$) for Subject {sub} in trial {trial}")
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Save the plot
        save_dir = os.path.join(output_dir, 'accuracy_barplot')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        plot_path = os.path.join(save_dir, f'encoding_accuracy_subj{format(sub, "02")}_barplot_trial_{trial}.png')
        plt.savefig(plot_path)
        print(f"Plot saved at: {plot_path}")
        trial += 1


# ==============================================================================
# Algorithm to plot the accuracy of the MLP model in fsavarage space on each ROI
# ==============================================================================

def mlp_accuracy_plot_fsavarage(sub, project_dir, output_dir):
    """
    Plots the encoding accuracy for each ROI in fsavarage space
    :param sub: The subject for which the results will be plotted
    :param project_dir: The project directory
    :param output_dir: The project outputs directory
    :return:
    """

    print(f'>>> Encoding Accuracy Barplot (fsaverage) for Subject {sub} <<<')

    # Load the noise-ceiling-normalized encoding accuracy
    data_dir = os.path.join(output_dir, 'encoding_accuracy')
    accuracy_files = sorted(glob.glob(os.path.join(data_dir, f'encoding_accuracy_subj{sub:02}_trial_*.npy')))

    trial = 0
    for accuracy_file in accuracy_files:
        data = np.load(accuracy_file, allow_pickle=True).item()
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
        plt.title(f'Noise-Ceiling-Normalized $R^2$ Scores for Subject {sub} (fsaverage) in trial {trial}')
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
        plt.savefig(os.path.join(save_dir, f'subj{sub:02}_fsaverage_r2_barplot_trial_{trial}.png'))
        trial +=1


# =========================================================================================================
# Algorithm that creates a barplot of the pearson correlation of the MLP for both hemispheres in each ROI
# =========================================================================================================

def mlp_plot_roi_correlation(sub, project_dir, output_dir, model_correlations):
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
                                                  'roi_masks', r), allow_pickle=True).item())
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
                                                      'roi_masks', lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(project_dir, 'Dataset', 'train_data', 'subj' + format(sub, '02'),
                                                      'roi_masks', rh_challenge_roi_files[r])))

    lh_correlations_list = model_correlations['lh_pearson_correlations']
    rh_correlations_list = model_correlations['rh_pearson_correlations']
    trial = 0
    for lh_correlations, rh_correlations in zip(lh_correlations_list, rh_correlations_list):
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
        plt.title(f'Model pearson correlation on each ROI for subject {sub}')
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
        save_dir = os.path.join(output_dir, 'correlation_barplot')
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, 'subj' + format(sub, '02') + f'_model_pearson_correlation_trial_{trial}.png'))
        trial +=1
