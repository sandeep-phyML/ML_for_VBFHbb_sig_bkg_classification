
This repository contains three scripts:

1. config.yml

* Contains all the file paths, tree names, branches, features, numbers, etc. used for DNN training.

2. utils.py
   This is a library where all necessary functions and classes are defined. It includes:

* Plot:

  * A collection of functions for:

    * TProfiles (correlation check)
    * Distribution of the DNN score
    * ROC curve

* PrepareDataset:

  * Reads the input ROOT files
  * Processes data (handles NaN values, normalization, etc.)
  * Calculates weights for training
  * Converts data into NumPy arrays or Pandas DataFrames
  * Prepares the dataset for training and prediction

* DNNModel:

  * Contains methods for DNN binary-class and multi-class models
  * Initialization, compilation, training
  * Saving models
  * Checking model performance (e.g., overfitting)
  * Prediction

* BasicMethods:

  * Holds all other functions that do not fit into the above categories

3. main.py

* This is the main script to run.
* Reads command-line arguments such as:

  * DNN or BDT training
  * Binary-class or multi-class training
  * Modes: only training, training with prediction, or only plotting/validation

Examples:

For only training a DNN binary model:
python main.py --train --biclass

For multi-class training and saving models:
python main.py --train

