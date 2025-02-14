"""
Python module to call the algonauts 2023 baseline linear regression model.
"""
import os.path

from Code.Linear_Regression_Model.baseline_encoding_model import \
    (baseline_encoding_extract_features, baseline_encoding_train_encoding_model, baseline_encoding_test_model,
     baseline_encoding_accuracy_plot, baseline_encoding_plot_roi_correlation, baseline_encoding_accuracy_plot_fsavarage)

# ================================================================================================
# Initialization
# ================================================================================================
project_dir = "../../" # Directory of the project
output_dir = "../../Outputs" # Directory in which the outputs will be stored
subjects = [1, 2, 3, 4, 5, 6, 7, 8] # Subjects that will be used during the execution of the program
# Directory in which the outputs of the model will be stored
linear_regression_models_dir = os.path.join(output_dir, 'Linear_regression_models')
# ================================================================================================
# Main function
# ================================================================================================
def use_algonauts_model(project_dir, output_dir):
    """""
    Utilizes the baseline model provided by the algonauts challenge
    :param project_dir:
        project directory
    :param output_dir:
        output directory
    """
    # Initialize local baseline model output directory
    local_baseline_encoding_dir = os.path.join(output_dir, 'baseline_encoding_model')
    os.makedirs(local_baseline_encoding_dir, exist_ok=True)
    for sub in subjects:
        # Extract features
        baseline_encoding_extract_features(sub, project_dir, local_baseline_encoding_dir)
        # Train model on both brain hemispheres
        baseline_encoding_train_encoding_model(sub, project_dir, output_dir)
        # Get the model pearson correlations
        model_correlations = baseline_encoding_test_model(sub, project_dir, local_baseline_encoding_dir)
        # Plot pearson correlation of the results with each ROI
        baseline_encoding_plot_roi_correlation(sub, project_dir, output_dir, model_correlations)
        # Plot R^2 score of the results for both brain hemispheres
        baseline_encoding_accuracy_plot(sub, project_dir, local_baseline_encoding_dir)
        # Plot R^2 score of the results for each ROI in fsavarage space
        baseline_encoding_accuracy_plot_fsavarage(sub, project_dir, local_baseline_encoding_dir)

use_algonauts_model(project_dir, linear_regression_models_dir) # Call main function



