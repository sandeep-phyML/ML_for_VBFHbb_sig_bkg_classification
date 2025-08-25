
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
* prepare even and odd datasets for training
    * convert the root tree data to pandas dataframe
    * Add any event filter for training , these are defined in the input_config.yml file , e.g events with T_btgb1 and T_btagb2 > 0.2605 will be used for training
    * add the label branch , will labels , e.g 0 for signal and 1 for background
    * by default it will reasmple the number to events to the number mentioned in config file ,# sole purpose of the resampling is to give equal importance to the signal and background when there is an assymetry
    * for weight training , get the weights branch or column , each will have the product of the corresponding weights , the name of the indivisual branches are defined in the config file
    * next normalise the weights with median scaling
    * now , combine all the pandas dataframes ( QCD , VBF , GGH etc ) , and suffle it ,
    * filter nan , infi etc with 0,
    * finally divide the data into odd and even datasets , based on the T_events ( event number ) and then return numpy array for features , weights and labels , also return mass array if required
* prepare datasets for the dnn score prediction 
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

