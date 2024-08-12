# TopologicalDataAnalysis3D
Code relative to 'Persistence Image from 3D Medical Image: Superpixel and Optimized Gaussian Coefficient'.

## Installation and Requirements
Use Anaconda to install the environment suitable for this code. Follow these steps:

1. Download and copy the `environment.yml` file to the target system.
2. Open the terminal (or Anaconda Prompt).
3. Create a new environment using the following command:
  ```bash
conda env create -f environment.yml
  ```

## Quick Start
To download the MedMNist dataset into the 'data' folder:
```bash
python DownloadMedMNist.py
```
To rename the folder of dataset:
```bash
python datafolderRename.py
```
To convert 3D medical images into persistence images:
```bash
python preprocess.py
```
If you need to select the directory to save the dataset, the number of superpixels, different homology groups, or different Optimized Gaussian Coefficients, please modify the corresonding parameters in 'preprocess.py'.

To train and predict:
```bash
python train.py
```
