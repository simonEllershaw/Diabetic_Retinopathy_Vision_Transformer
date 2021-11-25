# A Study of the Possible Advantages of Vision Transformers in the Automated Diagnosis of Diabetic Retinopathy from Fundus Images

TL;DR Repo to compare ViT vs CNN as diabetic retinopathy classifiers

This repo was written to perform a number of rapid experiments of finetuning and evaluation of vision transformers as diabetic retionopathy classifiers as part of my MRes at UCL.

Headline finding of this work was that ViT-S had a worse classification performance than a ResNet50 model on this task. Which is slightly against the current CV trend of transformer hype at the moment.

![Pre_Recall_Messidor.JPG](figures\Pre_Recall_Messidor.JPG?raw=true "Precision Recall Curve")

However, visualizing a transformer's attention is an interesting alternative approach to explainability than the saliency maps currently used by CNNs. 

![explain_vis.JPG](figures/explain_vis.JPG?raw=true "Explain Vis")

An extensive deep dive into the methodology, results and discussion of this project can be found in my MRes_Dissertation.pdf which is also in this repo.

## Setup

Download repo and run 

`pip install -r requirements.txt`

The models and datasets used are not contained in this repo and so must be downloaded from external sources. Note to run results.ipynb using precalculated metrics these steps are not required 

### Models

Models can be download from OneDrive. Then unzip and place in top level folder named models.
These are models finetuned from the excellent timm library.

### Datasets

Three separate datasets are used EyePACs (training), Messidor-1 (external validation) and IDRiD (explainability). These are all open-source and available at the below links. Although EyePACs is a v large dataset. 


These need to be downloaded extracted and then placed in the following file structure
- data
    - [eyePACs](kaggle.com/c/diabetic-retinopathy-detection/data). 
    - [messidor](https://www.adcis.net/en/third-party/messidor/)
    - [idrid](https://idrid.grand-challenge.org/)

Also eyepacs_gradability_grades.csv from [this repo](https://github.com/mikevoets/jama16-retina-replication/tree/master/vendor/eyepacs) is required to be placed in the eyePACs folder.

## Scripts

The scripts provided are in 2 categories
1. CLI interfaces to enable the training of models
    - preprocess.py: Sorts out file structures, squares and resizes images, creates explainability segmentation maps
    - train.py: Pytorch training of timm models
    - metrics.py: Calculates a range of model metrics and stores as a txt in the model's directory.
2. Notebooks to reproduce the metrics and figures presented in my thesis
    - data_visualistation.ipynb: Visualize input data and distribution
    - results.ipynb: Reproduce metrics and figures presented in thesis from model's metrics.txt

The rest of the code in the repo contains helper functions for these scripts. These were built for my own use and so the documentation is poor but all methods described in my thesis are implemented in this repo. If unclear please get in contact.