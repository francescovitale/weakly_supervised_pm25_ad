# Segmentation

The Segmentation folder contains both fault-free and faulty data, and segments these data through the data_segmentation.py into different time frames with multiple granularity settings (2.5%, 5%, 10%, 20%, 25%, and 50%).

# Calibration 

The Calibration folder contains different calibration implementations to apply to the segmented data and the ARPA reference. In particular, calibration.bat allows both the configuration of the environment and the calibration. It calls the calibration_single_models.py, which creates a calibration model for each time frame of the target granularity.

# Weakly-supervised labeling

The Weakly-Supervised Labeling folder contains the script to label the different PM25 windows within the time frames of the different datasets. The weakly_supervised_labeling.bat sets the environment and performs the labeling. It calls the weakly_supervised_labeling.py script, specifying the different parameters used to label the data (e.g., DBScan parameters and PCA variance threshold).

# Anomaly detection

The AnomalyDetection folder contains the Jupyter notebook "anomaly_detection_experiment.ipynb", which loads the datasets built through weakly-supervised labeling and carries out anomaly detection against the PM25 windows using three different widespread approaches for collective anomaly detection: LSTM-AE, GRU-AE and CNN-AE.

# Results

The Results folder contains the Calibration and Anomaly detection results, both in terms of their quantitative results under the Processing sub-folder and in terms of their visualization under the Visualization sub-folder.
