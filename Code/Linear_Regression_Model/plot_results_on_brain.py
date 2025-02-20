from copy import copy

import cortex
from matplotlib import pyplot as plt
import numpy as np
import os
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
    plt.savefig('algonauts_2023_challenge_winner_1.png', transparent=True, dpi=100)

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
    plt.savefig('algonauts_2023_full_surface_template.png', transparent=True, dpi=100)

    # Plot the challenge vertices surface --> ['PiYG', 'RdPu_r']
    data = np.append(lh_fsaverage, rh_fsaverage)
    idx = ~np.isnan(data)
    data[idx] = 5
    vertex_data = cortex.Vertex(data, subject, cmap='PiYG', vmin=0, vmax=100)
    cortex.quickshow(vertex_data, with_colorbar=False)
    plt.show()
    plt.savefig('algonauts_2023_challenge_vertices_surface.png', transparent=True, dpi=100)