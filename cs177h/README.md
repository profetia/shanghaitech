# VirFormer: Identifying viruses as an anomaly

Version: 1.0

Authors: Linshu Yang, Pengyu Long, Qingcheng Zhao


## Introduction

VirFormer is a transformer based model for virus nucleotide sequence prediction. It proposed a novel method to predict virus nucleotide sequences by identifying viruses as an anomaly. It is a two-stage model, which first uses an encoder to encode the nucleotide sequences into a latent space, and then uses a classifier to predict the virus nucleotide sequences. The encoder is a transformer based model, which can encode the nucleotide sequences into a latent space. The classifier is a cnn-based model, which can predict the virus nucleotide sequences by identifying viruses as an anomaly in the latent space.


## Dependencies

VirFormer requires Python 3.9 with the package of pytorch, numpy, torchmetrics, scikit-learn, jupyter and Biopython. We recommend the use Ananconda to install all dependencies.



## Installation

After installing dependencies, you can run the main.py in [encoder](encoder) to train the encoder. And you can run main.py, main_new.py, main_linear.py in [classifier](classifier) to train the classifiers. 

For testing, you can run auroc.ipynb in [classifier](classifier) to test different models.



## Dataset

Before you run our code, you need to download the data from NCBI using the Accession ID in [dataset](dataset). And you are recommended to divide them into human, prokaryote, simulation_abundance and virus folders. You may need to change some paths in our code to match the path of your dataset.

NCBI Accessions: https://epan.shanghaitech.edu.cn/l/wFKC6o

## Training

Our training requires at least 360GB memory and takes 36h to achieve barely convergence on 8 x Nvidia V100. 

## Misc

Presentation and paper can be found in [media](media). Details of the model can be found in [paper](media/final_report.pdf).

