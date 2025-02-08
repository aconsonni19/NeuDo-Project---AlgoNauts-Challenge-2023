README.txt
# =============================================================================
# General data information
# =============================================================================
The challenge data comes from the Natural Scenes Dataset (NSD) (Allen et al.,
2022), a massive 8-subjects dataset of high-quality 7T fMRI responses to images
of natural scenes coming from the COCO database (Lin et al., 2014).

During the NSD experiment each subject viewed 10,000 distinct images, and a
special set of 1,000 images was shared across subjects (eight participants ×
9,000 unique images + 1,000 shared images = 73,000 images). Each of the 10,000
images was presented three times, for a total of 30,000 image trials per
subject. Subjects were instructed to focus on a fixation cross at the center of
the screen and performed a continuous recognition task in which they reported
whether the current image had been presented at any previous point in the
experiment.

For every subject the NSD experiment was split across 40 scan sessions: not all
subjects completed all of them, resulting in different amounts of recorded data
between subjects. The fMRI data of the last three sessions of every subject is
withheld and constitutes the basis for the test split of the 2023 Algonauts
Project challenge, whereas the data of the remaining sessions is released and
constitutes the basis for the training split of the challenge.

The challenge uses preprocessed fMRI responses (BOLD response amplitudes) from
each subject that have been projected onto a common cortical surface group
template (FreeSurfer's fsaverage surface). Brain surfaces are composed of
vertices, and the challenge data consists of a subset of cortical surface
vertices in the visual cortex (a region of the brain specialized in processing
visual input) that were maximally responsive to visual stimulation. We provide
the data in right and left hemispheres. Further information on NSD acquisition
and preprocessing is provided on the challenge website and in the NSD paper.




# =============================================================================
# Data directory structure
# =============================================================================
The Algonauts 2023 challenge data is organized into the following directory
structure:

	_______________________________________________
	../algonauts_2023_challenge_data/
	│
	└───subj01/
	│	│
	│	└───roi_masks/
	│	│	└───lh.all-vertices_fsaverage_space.npy
	│	│	└───lh.floc-bodies_challenge_space.npy
	│	│	└───lh.floc-bodies_fsaverage_space.npy
	│	│	...
	│	│	└───mapping_floc-bodies.npy
	│	│	...
	│	│	└───rh.streams_fsaverage_space.npy
	│	│
	│	└───test_split/
	│	│	│
	│	│	└───test_images/
	│	│		└───test-0001_nsd-00845.png
	│	│		└───test-0002_nsd-00946.png
	│	│		...
	│	│		└───test-0159_nsd-72547.png
	│	│
	│	└───training_split/
	│		│
	│		└───training_fmri/
	│		│	└───lh_training_fmri.npy
	│		│	└───rh_training_fmri.npy
	│		│
	│		└───training_images/
	│			└───train-0001_nsd-00013.png
	│			└───train-0002_nsd-00027.png
	│			...
	│			└───train-9841_nsd-72999.png
	│
	...
	│
	└───subj08/
		│
		...
	‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

The directory structure of subjects 2 to 8 is analogous to the the one of
subject 1.

The following sections provide detailed information on the content of the
folders "../training_split/", "../test_split/" and "../roi_masks/".




# =============================================================================
# Training split
# =============================================================================
The "../training_split/" folder includes two subfolders: "../training_fmri/"
and "../training_images/".

"../training_images/" :	this folder contains the training images as ".png"
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾	files. For each of the 8 subjects there are [9841,
 						9841, 9082, 8779, 9841, 9082, 9841, 8779] different
 						training images. As an example, the first training
 						image of subject 1 is named "train-0001_nsd-00013.png".
						The first index ("train-0001") orders the images so to
						match the image conditions dimension of the fMRI
						training split data. This indexing starts from 1. The
						second index ("nsd-00013") corresponds to the 73,000 NSD
						image IDs that you can use to map the image back to the
						original ".hdf5" NSD image file (which contains all the
						73,000 images used in the NSD experiment), and from
						there to the COCO dataset images for metadata). The
						73,000 NSD images IDs in the filename start from 0, so
						that you can directly use them for indexing the ".hdf5"
						NSD images in Python. Note that the images used in the
						NSD experiment (and here in the Algonauts 2023
						challenge) are cropped versions of the original COCO
						images. Therefore, if you wish to use the COCO image
						metadata you first need to adapt it to the cropped image
						coordinates. You can find code to perform this operation
						in the "Useful links" section at the end of this
						document.

"../training_fmri/"	:	this folder contains the fMRI responses to the training
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾		images. The fMRI data is shared through two ".npy"
						files, corresponding to the neural responses of the
						left hemisphere ("lh_training_fmri.npy") and right
						hemisphere ("rh_training_fmri.npy") to the training
						images. The fMRI data is z-scored within each NSD scan
						session and averaged across image repeats, resulting in
						2D arrays with the number of training images as rows and
						the amount of brain surface vertices as columns. The
						left (LH) and right (RH) hemisphere files consist of,
						respectively, 19,004 and 20,544 vertices, with the
						exception of subjects 6 (18,978 LH and 20,220 RH
						vertices) and 8 (18,981 LH and 20,530 RH vertices) due
						to missing data.




# =============================================================================
# Test split
# =============================================================================
The "../test_split/" folder includes one subfolders: "../test_images/". The
corresponding fMRI visual responses are not released.

"../test_images/" :	this folder contains the test images as ".png" files. For
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾	each of the 8 subjects there are [159, 159, 293, 395, 159,
					293, 159, 395] different images. The file naming scheme is
					the same as for the train images.




# =============================================================================
# ROI masks
# =============================================================================
The visual cortex is divided into multiple areas having different functional
properties, referred to as regions-of-interest (ROIs). The "../roi_masks/"
folder contains the indices for selecting the vertices corresponding to specific
ROIs. Challenge participants can optionally use these ROI indices at their own
discretion (e.g., to build different encoding models for functionally different
regions of the visual cortex). However, the challenge evaluation metric is
computed over all available vertices, not over any single ROI.For the ROI
definition please see the NSD data manual (linked in the "Useful links" section
at the end of this document). Note that there are separate files for the left
("lh.*.npy") and right ("rh.*.npy") hemispheres, and that not all ROIs exist in
all hemispheres and subjects.

The ROIs are divided into different classes:
"prf-visualrois"	:	early retinotopic visual regions [V1v, V1d, V2v V2d,
						V3v, V3d, hV4].
"floc-bodies"		:	body-selective regions [EBA, FBA-1, FBA-2, mTL-bodies].
"floc-faces"		:	face-selective regions [OFA, FFA-1, FFA-2, mTL-faces,
						aTL-faces].
"floc-places"		:	place-selective regions [OPA, PPA, RSC].
"floc-words"		:	word-selective regions [OWFA, VWFA-1, VWFA-2, mfs-words,
						mTL-words].
"streams"			:	early, intermediate, and high-level streams along the
						dorsal, lateral, and ventral directions [early,
						midventral, midlateral, midparietal, ventral, lateral,
						parietal].

The ROI indices are shared in both challenge space (e.g.,
"lh.floc-bodies_challenge_space.npy") and fsaverage space (e.g.,
"lh.floc-bodies_fsaverage_space.npy"). The challenge space is based on the
vertices used in the challenge and it changes between subjects and hemispheres
(since subjects and hemispheres differ in amounts of vertices). The fsaverage
space is based on a brain surface template common for all subjects (FreeSurfer's
fsaverage surface), which includes vertices across the entire cortex (163,842
vertices per hemispheres, so more than the challenge vertices). The challenge
space ROI indices allow you to select the vertices belonging to a specific ROI
among all the challenge vertices, and the fsaverage space ROI indices allow you
to map challenge space vertices to fsaverage space to plot data on brain surface
plots. The Colab challenge tutorial provides examples on how to perform these
operations.

The "*h.*_*_space.npy" files contain the indices (in challenge space or
fsaverage space) of all ROIs within a given ROI class. Each ROI class uses
different integer values to label the vertices belonging to the different ROIs
within it. The "mapping_*.npy" files are Python dictionaries that allow you to
map the different integer values to the corresponding ROI names. The Colab
challenge tutorial provides examples on how to use the mapping dictionaries to
index the vertices of a given ROI of interest.

The files "*h.all-vertices_fsaverage_space.npy" corresponds to binary indices of
all vertices used in the challenge mapped to fsaverage space (represented as
ones). We do not provide the indices of all vertices in challenge space, since
that would simply correspond to all available data.




# =============================================================================
# Useful links
# =============================================================================
Challenge Website:
	http://algonauts.csail.mit.edu/

Challenge Paper:
	https://arxiv.org/abs/2301.03198

Challenge Data:
	https://docs.google.com/forms/d/e/1FAIpQLSehZkqZOUNk18uTjRTuLj7UYmRGz-OkdsU25AyO3Wm6iAb0VA/viewform?usp=sf_link

Challenge DevKit Tutorial (Colab):
	https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link

Challenge DevKit Tutorial (GitHub):
	https://github.com/gifale95/algonauts_2023

Challenge Overview and Colab Tutorial Walkthrough Video:
	https://youtu.be/KlwSDpxUX6k

Challenge Submission Page (CodaLab):
	https://codalab.lisn.upsaclay.fr/competitions/9304

CodaLab Submission Walkthrough Video:
	https://youtu.be/6b8OuMSXIpA

Natural Scenes Dataset (NSD):
	https://naturalscenesdataset.org/

NSD Data Manual:
	https://cvnlab.slite.page/p/CT9Fwl4_hc/NSD-Data-Manual

COCO Database:
	https://cocodataset.org/

Adapt COCO Metadata to Cropped NSD Images:
	https://github.com/styvesg/nsd_gnet8x/blob/main/data_preparation.ipynb




# =============================================================================
# Paper and Citations
# =============================================================================
If you use the data provided for the Algonauts Project 2023 Challenge please
cite the following papers:

1. Gifford AT, Lahner B, Saba-Sadiya S, Vilas MG, Lascelles A, Oliva A, Kay K,
	Roig G, Cichy RM. 2023. The Algonauts Project 2023 Challenge: How the Human
	Brain Makes Sense of Natural Scenes. arXiv preprint, arXiv:2301.03198.
	DOI: https://doi.org/10.48550/arXiv.2301.03198

2. Allen EJ, St-Yves G, Wu Y, Breedlove JL, Prince JS, Dowdle LT, Nau M,
	Caron B, Pestilli F, Charest I, Hutchinson JB, Naselaris T, Kay K. 2022.
	A massive 7T fMRI dataset to bridge cognitive neuroscience and computational
	intelligence. Nature Neuroscience, 25(1):116–126.
	DOI: https://doi.org/10.1038/s41593-021-00962-x



