# Reconstruction from spatio-spectrally coded multispectral light fields

This repository contains the **code** and **model weights** for the paper

Schambach, Maximilian, Jiayang Shi, and Michael Heizmann:   
"Spectral Reconstruction and Disparity from Spatio-Spectrally Coded Light Fields via Multi-Task Deep Learning."   
In: International Conference on 3D Vision (3DV), 2021.  
https://doi.org/10.1109/3DV53792.2021.00029

with additional results from my thesis

Schambach, Maximilian:  
"Reconstruction from Spatio-Spectrally Coded Multispectral Light Fields."  
PhD Thesis, Karlsruhe Institute of Technology, 2022.  
https://doi.org/10.5445/IR/1000143731  
https://maxschambach.github.io/thesis

if you use this work, please cite the paper and/or the thesis.

## Used data
The data used to train, validate and test the model can be downloaded from IEEE DataPort here:
https://ieee-dataport.org/open-access/multispectral-light-field-dataset-light-field-deep-learning

Throughout, the `DATASET_MULTISPECTRAL_PATCHED_F_9x9_36x36.zip` is used.
It provides a test, train, and validation dataset.

## Create environment
To train and test models, create a new conda environment as provided
```bash
conda env create -n lf-reconstruct -f environment.yml
conda activate lf-reconstruct
```
All experiments in the paper and my thesis were conducted with `tensorflow-gpu==2.4` and Python 3.9 as specified in the `environment.yml`. 
Feel free to train new models with up-to-date versions.

## Pretrained model test
You find the weights of a selection of experiments from the paper and thesis in the `weights` folder.
You can use the `test.py` script to validate the results on the test dataset (`test.h5`).
Please adapt the paths to the used model weights and the test dataset.

## Available weights
The notation of the training strategy follows my thesis. 
In particular, the best training strategy `MT Uncertaintyy AL NormGradSim` uses adaptive multi task loss weighting using uncertainty and an adaptive auxiliary loss weighting using normalized gradient similarity.
In the paper, this was denoted as `MTU + AL`.

If not noted otherwise, models correspond to the network based on 3D convolutions and random coding masks (as presented in the paper) using an angular resolution of 9x9. Better performing models, either using regular coding masks or based on 4D separable convolutions, which were investigated in my thesis are listed below.

| Experiment | Weights  | 
|----|----| 
| MT Uncertainty AL NormGradSim | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim.h5) | 
| ST Central AL NormGradSim     | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/st_central_al_normgradsim.h5)   |
| ST Disparity AL NormGradSim   | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/st_disp_al_normgradsim.h5)      |

### Different angular resolutions
All models trained with random coding masks and `MT Uncertainty AL NormGradSim` training strategy

| Angular Resolution | Weights  | 
|--------------------|----| 
| 3x3                | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim_3d_3x3.h5) | 
| 5x5                | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim_3d_5x5.h5) | 
| 7x7                | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim_3d_7x7.h5) | 
| 9x9                | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim_3d_9x9.h5) | 

### Using different coding masks during training
All models trained with 9x9 angular resolution and `MT Uncertainty AL NormGradSim` training strategy.

| Coding Mask       | Weights  | 
|-------------------|----| 
| random            | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim.h5) | 
| random macropixel | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim_random_macropx.h5) | 
| regular           | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim_regular.h5) | 
| regular optimized | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim_regular_opt.h5) | 

### Model based on 4D separable convolution
Trained with 9x9 angular resolution, random coding mask, and `MT Uncertainty AL NormGradSim` training strategy.

| Experiment                    | Weights  | 
|-------------------------------|----| 
| MT Uncertainty AL NormGradSim | [download](https://gitlab.com/MaxSchambach/reconstruction-from-spectrally-coded-light-fields/-/blob/main/weights/mt_uncertainty_al_normgradsim_4d.h5) | 

## Training from scratch
You can use the train and test data (see above) to train the models from scratch.
To do so, use the provided `train.py` script. 
Fill in the paths of the train, validation, and test dataset used.
The parameters set corresponds to the default parameters used.
You can of course adapt them to your own needs.
See the supplementary of the paper or my thesis for details on all hyperparameters.
You can find a more complex but flexible train and validation setup with all conducted experiments in an [additional repo](https://gitlab.com/MaxSchambach/thesis-experiments).