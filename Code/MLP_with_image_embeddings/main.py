"""

"""


import os
import torch
from mlp_train_and_test import (extract_clip_features, train_mlp_models, test_mlp_encoding, mlp_accuarcy_plot,
                                mlp_accuracy_plot_fsavarage, mlp_plot_roi_correlation)




clip_models = ['ViT-B/32', 'ViT-L/14', 'RN50']
device = "cuda" if torch.cuda.is_available() else "cpu"


### Local dirs
project_dir = "../../"
output_dir = "../../Outputs"
subjects = [1, 2, 3, 4, 5, 6, 7, 8]
mlp_model_output_dir = os.path.join(output_dir, 'MLP_with_clip')


if not os.path.isdir(mlp_model_output_dir):
    os.makedirs(mlp_model_output_dir)

for model in clip_models:
    print(f">>> Using CLIP model {model}")
    model_output_dir = os.path.join(mlp_model_output_dir, model)
    os.makedirs(model_output_dir, exist_ok=True)
    # for sub in subjects:
    # extract_clip_features(1, model, project_dir, model_output_dir)
    # train_mlp_models(1, project_dir, model_output_dir)
    model_dir = os.path.join(mlp_model_output_dir, model)
    subdirs = [x[1] for x in os.walk(model_dir)][0]
    subdirs.remove("clip_features")  # Remove unwanted dir
    for subdir in subdirs:
        print(f'Analyzing subdir {subdir}')
        model_correlations = test_mlp_encoding(1, project_dir, os.path.join(mlp_model_output_dir, model, subdir))
        mlp_plot_roi_correlation(1, project_dir, os.path.join(mlp_model_output_dir, model, subdir),
                                 model_correlations)
        mlp_accuarcy_plot(1, project_dir, os.path.join(mlp_model_output_dir, model, subdir))
        mlp_accuracy_plot_fsavarage(1, project_dir, os.path.join(mlp_model_output_dir, model, subdir))


#train_mlp(1, project_dir, mlp_model_output_dir, device)










