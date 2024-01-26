# drm_sequence_net

This code is for the paper "Autoregressive Models for Crystallographic Orientation Prediction in the Presence of Symmetries."

Paper can be found [here](https://github.com/arunpatala/drm_sequence_net/blob/main/Predicting_Orientations_with_Symmetries.pdf)

Most of the code is inspired by [this repository](https://github.com/MalloryWittwer/drm_ml_demo). The EulerNet results can be viewed there.

#### Test

Run `main.py` to obtain the following results of the SequenceNet. Test Time augmentation results are also noted below. The model and the data will be downloaded automatically.

| Metric | Value |
|--------|-------|
| Test misorientation median | 4.74 |
| Test misorientation mean | 8.24 |
| TTA misorientation median | 3.96 |
| TTA misorientation mean | 6.70 |

#### Training

Run `train.py` to train the sequence net model with the default settings. 
Using CUDA with a GPU is recommended.

##### Data

Please download the **data** folder (3.5 GB) from the Mendeley Dataset available at [DOI:10.17632/z8bh7n5b7d.1](https://data.mendeley.com/datasets/z8bh7n5b7d/1). Copy the data folder into the root directory of the repository.

Or run `main.py` to download the data.

#### Data Description

The **data** folder contains:

(i) All training and evaluation sets used to derive the results. 

(ii) Three additional files: 
- **/samples/08/drm_data.npy**: A 4D numerical matrix (shape (x, y, theta, phi), type uint8) representing the experimental DRM dataset of the test specimen showcased in Figure 3 of the paper.
- **/samples/08/eulers.npy**: The corresponding matrix of Euler angles measured by EBSD for this test specimen (shape (x, y, 3), type float32).
- **anomaly_specimen.npy**: The DRM dataset of the specimen shown in Figure 6 of the paper to demonstrate the detection of out-of-distribution data.


