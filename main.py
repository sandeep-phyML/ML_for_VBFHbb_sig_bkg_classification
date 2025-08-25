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
parser.add_argument('--plot_var_distribution', action='store_true', help="is plot the variable distributions ? ")
parser.add_argument('--plot_roc', action='store_true', help="is plot the roc curve ? ")
parser.add_argument('--plot_tprofile', action='store_true', help="is plot the tProfile ? ")
args = parser.parse_args()

# read the config file
basic_tool = BasicMethods()
config_file_name = "input_config.yml"
config = basic_tool.read_config_file(config_file_name)
log_path = os.path.join(config['output_log']['folder_path'],config['output_log']['file_name'])
# basic_tool.create_log_folder(log_path)
if args.prepare_5perc: # prepare 5% of the data for training if have not prepared it yet
    dataset_ = PrepareDataset(  config = config ,output_log_path = log_path , is_biclass = args.biclass)
    dataset_.create_five_perc_data( config["full_data_path"] )

if args.process_data:  # only for external perposes like , training a BDT , for DNN no need to use this feature 
    dataset_ = PrepareDataset( config = config ,output_log_path = log_path , is_biclass = args.biclass)
    for file in config["prediction_files"]["file_names"]: # clean the data , replace nan with zero and apply medium btag 
        file_path = os.path.join(config["prediction_files"]["folder_path"], file)
        dataset_.filter_nan_with_zero_event_sel( file_path ,["T_btgb1","T_btgb2"],[0.260,0.260])
    for file in config["train_mclass_files_labels"]["file_names_labels"]:
        file_path = os.path.join(config["train_mclass_files_labels"]["folder_path"], file["file_name"])
        output_file_path = os.path.join("/Users/sandeeppradhan/Desktop/VBF_Analysis_Folder/2022_pre_EE_Ntuples/BDT_Train_Inputs_Ntuples/", file["file_name"])
        dataset_.prepare_data_for_bdt_training( file_path, output_file_path,args.nresample,["T_btgb1","T_btgb2"],[0.260,0.260] , "train_weight"  )
    
# prepare the dataset for training
if args.train :
    print("Preparing dataset for training ")
    dataset_ = PrepareDataset(  config = config ,output_log_path = log_path , is_biclass = args.biclass)
    train_odd, train_even , weight_odd , weight_even , label_odd , label_even , mass_odd , mass_even = dataset_.prepare_even_odd_datasets_train()
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
    model_.compile_dnn_model()
    model_.train_dnn_model()
    model_.save_model_weights()

# predict DNN score model 
if args.predict:
    dataset_ = PrepareDataset(  config = config ,output_log_path = log_path , is_biclass = args.biclass)
    model_ = DNNModel(config,log_path, is_biclass=args.biclass, is_bdt=args.bdt)
    model_.pred_data_dict = dataset_.prepare_even_odd_datasets_prediction() # read all the files , from config and prepare even odd dataset , clean nan with zero etc 
    model_.predict_dnn_score()

# show results and save plots
plot_ = Plot(config, log_path)
if args.plot_var_distribution:
    plot_.plot_var_distribution(["DNN_BiClass"])
    plot_.plot_var_distribution(["BiClassANN"])
    plot_.plot_var_distribution(["DNN_VBF","DNN_QCD"])
    plot_.plot_var_distribution(["DNN_VBF","DNN_QCD","DNN_TT"])
    plot_.plot_var_distribution(["DNN_VBF","DNN_GGH","DNN_QCD"])
    plot_.plot_var_distribution(["DNN_QCD","DNN_VBF"])
    #plot_.plot_var_distribution(["DNN_GGH","DNN_QCD"," DNN_VBF"])
    plot_.plot_var_distribution(["DNN_GGH","DNN_VBF"])
    plot_.plot_var_distribution(["DNN_Z2Q","DNN_QCD"])
    plot_.plot_var_distribution(["DNN_Z2Q","DNN_VBF"])
    # plot_.plot_var_distribution(["DNN_Z2Q","DNN_QCD","DNN_TT"])
    # plot_.plot_var_distribution(["DNN_GGH","DNN_QCD","DNN_TT"])
if args.roc:
    plot_.plot_roc_curve(["DNN_BiClass"])
    #plot_.plot_roc_curve(["BiClassANN"])
    plot_.plot_roc_curve(["DNN_VBF","DNN_QCD"])
    plot_.plot_roc_curve(["DNN_VBF","DNN_QCD","DNN_TT"])
if args.plot_tprofile:
    file_path = os.path.join(config["prediction_files"]["folder_path"], "tree_VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8_2022_btgsf.root")
    plot_.plot_tprofile(file_path ,"output_plot_models/tprofile_biclass_vbf.png", is_mclass = False ,is_data = False,nbins = 50,mbb_range=(110.0, 140.0),vbfbclass="DNN_BiClass",vbf_dnn = "DNN_VBF",qcd_dnn = "DNN_QCD")
    plot_.plot_tprofile(file_path ,"output_plot_models/tprofile_mclass_vbf.png", is_mclass = True ,is_data = False,nbins = 50,mbb_range=(110.0, 140.0),vbfbclass="DNN_BiClass",vbf_dnn = "DNN_VBF",qcd_dnn = "DNN_QCD")
    plot_.plot_tprofile(file_path ,"output_plot_models/tprofile_biclass_vbf_qcd.png", is_mclass = False ,is_data = False,nbins = 50,mbb_range=(110.0, 140.0),vbfbclass="BiClassANN",vbf_dnn = "DNN_VBF",qcd_dnn = "DNN_QCD")
    file_path = os.path.join(config["prediction_files"]["folder_path"], "tree_JetMET_2022_btgsf.root")
    plot_.plot_tprofile(file_path ,"output_plot_models/tprofile_biclass_data.png", is_mclass = False ,is_data = True,nbins = 50,mbb_range=(110.0, 140.0),vbfbclass="DNN_BiClass",vbf_dnn = "DNN_VBF",qcd_dnn = "DNN_QCD")
    plot_.plot_tprofile(file_path ,"output_plot_models/tprofile_mclass_data.png", is_mclass = True ,is_data = True,nbins = 50,mbb_range=(110.0, 140.0),vbfbclass="DNN_BiClass",vbf_dnn = "DNN_VBF",qcd_dnn = "DNN_QCD")
    plot_.plot_tprofile(file_path ,"output_plot_models/tprofile_biclass_vbf_qcd.png", is_mclass = False ,is_data = True,nbins = 50,mbb_range=(110.0, 140.0),vbfbclass="BiClassANN",vbf_dnn = "DNN_VBF",qcd_dnn = "DNN_QCD")




