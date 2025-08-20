#import ncecessary libraries
import os
from utils import Plot , PrepareDataset , DNNModel , BasicMethods
import argparse

# get arguments from command line
parser = argparse.ArgumentParser(description="Model pipeline options")
parser.add_argument('--biclass', action='store_true', help="biclass / multiclass ")
parser.add_argument('--bdt', action='store_true', help="bdt or DNN ")
parser.add_argument('--train', action='store_true', help="Run training")
parser.add_argument('--predict', action='store_true', help="Run prediction")
parser.add_argument('--results', action='store_true', help="Show/save results")
parser.add_argument('--process_data', action='store_true', help="prepare data for training and prediction bdt etc ")
parser.add_argument('--nresample', type=int,default=100000, help="Number of resamples to perform (default: 100000)")
parser.add_argument('--prepare_5perc', action='store_true', help="prepare 5% of the data for training ")
args = parser.parse_args()

# read the config file
basic_tool = BasicMethods()
config_file_name = "input_config.yml"
config = basic_tool.read_config_file(config_file_name)
log_path = os.path.join(config['output_log']['folder_path'],config['output_log']['file_name'])
basic_tool.create_log_folder(log_path)
if args.prepare_5perc:
    dataset_ = PrepareDataset(  config = config ,output_log_path = log_path , is_biclass = args.biclass)
    dataset_.create_five_perc_data( config["full_data_path"] )

if args.process_data:
    dataset_ = PrepareDataset(  config = config ,output_log_path = log_path , is_biclass = args.biclass)
    # for file in config["prediction_files"]["file_names"]: # clean the data , replace nan with zero and apply medium btag 
    #     file_path = os.path.join(config["prediction_files"]["folder_path"], file)
    #     dataset_.filter_nan_with_zero_event_sel( file_path ,["T_btgb1","T_btgb2"],[0.260,0.260])
    for file in config["train_mclass_files_labels"]["file_names_labels"]:
        file_path = os.path.join(config["train_mclass_files_labels"]["folder_path"], file["file_name"])
        output_file_path = os.path.join("/Users/sandeeppradhan/Desktop/VBF_Analysis_Folder/2022_pre_EE_Ntuples/BDT_Train_Inputs_Ntuples/", file["file_name"])
        dataset_.prepare_data_for_bdt_training( file_path, output_file_path,args.nresample,["T_btgb1","T_btgb2"],[0.260,0.260] , "train_weight"  )
    
# prepare the dataset for training
if args.train or args.predict:
    print("Preparing dataset for training or prediction...")
    dataset_ = PrepareDataset(  config = config ,output_log_path = log_path , is_biclass = args.biclass)
    train_odd, train_even , weight_odd , weight_even , label_odd , label_even , mass_odd , mass_even = dataset_.get_np_feaweilabel_odd_even_train()
    print(train_odd.shape, label_odd.shape, weight_odd.shape)
    print(train_even.shape, label_even.shape, weight_even.shape)

    # Load and configure the training feature data , labels and weights the DNN models
    model_ = DNNModel(config,log_path, is_biclass=args.biclass, is_bdt=args.bdt)
    model_.features_odd_data = train_odd
    model_.features_even_data = train_even
    model_.labels_odd_data = label_odd
    model_.labels_even_data = label_even
    model_.weights_odd_data = weight_odd
    model_.weights_even_data = weight_even

# compile , train and save the model weights
if args.train:
    model_.compile_dnn_model()
    model_.train_dnn_model()
    model_.save_model_weights()
'''
# predict DNN score model 
if args.predict:
    model_.pred_data_dict = dataset_.prepare_pred_data()
    model_.predict_mclass_model()

# show results and save plots
if args.results:
    plot_ = Plot(config, log_path)
    plot_.plot_mclass_results(model_.pred_data_dict, model_.model_weights_path)
    plot_.save_mclass_results(model_.pred_data_dict, model_.model_weights_path)
    plot_.plot_mclass_loss(model_.loss_history, model_.model_weights_path)
    plot_.save_mclass_loss(model_.loss_history, model_.model_weights_path)
'''

