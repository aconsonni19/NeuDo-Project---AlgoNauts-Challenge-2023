"""
Python script to train and test the MLP architecture based on CLIP
"""
import os
from mlp_train_and_test import (extract_clip_features, train_mlp_models, test_mlp_encoding, mlp_accuarcy_plot,
                                mlp_accuracy_plot_fsavarage, mlp_plot_roi_correlation)

# =============================================================================
# Initialization
# =============================================================================

clip_models = ['ViT-B/32', 'RN50'] # CLIP models used for extracting features


# Local directories
project_dir = "../../"
output_dir = "../../Outputs"
subjects = [5,6,7,8]
mlp_model_output_dir = os.path.join(output_dir, 'MLP_with_clip')
os.makedirs(mlp_model_output_dir, exist_ok=True)

# =============================================================================
# Main Loop
# =============================================================================
for model in clip_models:
    print(f">>> Using CLIP model {model}")
    # Set the output directory of the model
    model_output_dir = os.path.join(mlp_model_output_dir, model)
    os.makedirs(model_output_dir, exist_ok=True)
    for sub in subjects:
        # Extract features with the current clip model
        extract_clip_features(sub, model, project_dir, model_output_dir)
        # Train the MLP on the features extracted; the model's hyperparameters will be set using random search
        train_mlp_models(sub, project_dir, model_output_dir, 5)
        # Gett all models directories
        model_dir = os.path.join(mlp_model_output_dir, model)
        subdirs = [x[1] for x in os.walk(model_dir)][0]
        subdirs.remove("clip_features")  # Remove unwanted dir
        for subdir in subdirs:
            print(f'Analyzing subdir {subdir}')
            # Calculate pearson correlation for the model's results
            model_correlations = test_mlp_encoding(sub, project_dir, os.path.join(mlp_model_output_dir, model, subdir))
            # Plot model's pearson correlation on every ROI for both hemispheres
            mlp_plot_roi_correlation(sub, project_dir, os.path.join(mlp_model_output_dir, model, subdir),
                                     model_correlations)
            # Plot model's accuracy (R^2 score) for both hemispheres
            mlp_accuarcy_plot(sub, project_dir, os.path.join(mlp_model_output_dir, model, subdir))
            # Plot model's accuracy (R^2 score) on every ROI for both hemispheres (in fsaverage space)
            mlp_accuracy_plot_fsavarage(sub, project_dir, os.path.join(mlp_model_output_dir, model, subdir))












