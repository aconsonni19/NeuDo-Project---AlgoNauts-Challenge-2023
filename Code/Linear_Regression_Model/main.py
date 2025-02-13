# ================================================================================================
# Python module to call the various linear regression models contained in Linear_Regression_Model
# ================================================================================================
import os.path

from Code.Linear_Regression_Model.baseline_encoding_model import \
    (baseline_encoding_extract_features, baseline_encoding_train_encoding_model, baseline_encoding_test_model,
     baseline_encoding_accuracy_plot, baseline_encoding_plot_roi_correlation, baseline_encoding_accuracy_plot_fsavarage)


### Local dirs
project_dir = "../../"
output_dir = "../../Outputs"
subjects = [1, 2, 3, 4, 5, 6, 7, 8]
linear_regression_models_dir = os.path.join(output_dir, 'Linear_regression_models')

def use_algonauts_model(project_dir, output_dir):
    """""
    Utilizes the baseline model provided by the algonauts challenge
    :param project_dir:
        project directory
    :param output_dir:
        output directory
    """
    local_baseline_encoding_dir = os.path.join(output_dir, 'baseline_encoding_model')
    if os.path.isdir(local_baseline_encoding_dir) == False:
        os.makedirs(local_baseline_encoding_dir)
    for sub in subjects:
        baseline_encoding_extract_features(sub, project_dir, local_baseline_encoding_dir) # Extract features
        baseline_encoding_train_encoding_model(sub, project_dir, output_dir) # Train model on both brain hemispheres
        model_correlations = baseline_encoding_test_model(sub, project_dir, local_baseline_encoding_dir)
        baseline_encoding_plot_roi_correlation(sub, project_dir, output_dir, model_correlations)
        baseline_encoding_accuracy_plot(sub, project_dir, local_baseline_encoding_dir)
        baseline_encoding_accuracy_plot_fsavarage(sub, project_dir, local_baseline_encoding_dir)


use_algonauts_model(project_dir, linear_regression_models_dir)



