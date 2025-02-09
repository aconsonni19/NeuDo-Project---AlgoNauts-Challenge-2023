# ================================================================================================
# Python module to call the various linear regression models contained in Linear_Regression_Models
# ================================================================================================
import os.path

from Code.Linear_Regression_Models.baseline_encoding_model import \
    baseline_encoding_extract_features, baseline_encoding_train_encoding_model

### Local dirs
project_dir = "../../"
output_dir = "../../Outputs"
linear_regression_models_dir = os.path.join(output_dir, 'Linear_regression_models')


def use_algonauts_model(project_dir, output_dir):
    """""
    Utilizes the model contain in algonauts2023_baseline_model
    :param sub:
        Subject used
    :param project_dir:
        project directory
    :param output_dir:
        output directory
    """
    local_baseline_encoding_dir = os.path.join(output_dir, 'baseline_encoding_model')
    if os.path.isdir(local_baseline_encoding_dir) == False:
        os.makedirs(local_baseline_encoding_dir)

    for sub in range (1, 9):
        # baseline_encoding_extract_features(sub, project_dir, local_baseline_encoding_dir) # Extract features
        baseline_encoding_train_encoding_model(sub, project_dir, output_dir) # Train model on both brain hemispheres






use_algonauts_model(project_dir, linear_regression_models_dir)

