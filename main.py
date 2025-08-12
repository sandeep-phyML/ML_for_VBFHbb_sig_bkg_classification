#import ncecessary libraries
import os
from utils import Plot , PrepareDataset , DNNModel , DNNModelTraining , BasicMethods
import argparse

# get arguments from command line
parser = argparse.ArgumentParser(description="Model pipeline options")
parser.add_argument('--train', action='store_true', help="Run training")
parser.add_argument('--predict', action='store_true', help="Run prediction")
parser.add_argument('--results', action='store_true', help="Show/save results")
args = parser.parse_args()

# read the config file
basic_tool = BasicMethods()
config_file_name = "input_config.yml"
config = basic_tool.read_config_file(config_file_name)
log_path = os.path.join(config['output_log']['folder_path'],config['output_log']['file_name'])
basic_tool.create_log_folder(log_path)

# prepare the dataset for training
dataset_ = PrepareDataset(  config = config ,output_log_path = log_path)
train_odd, train_even , weight_odd , weight_even , label_odd , label_even , mass_odd , mass_even = dataset_.get_np_feaweilabel_odd_even_train()
print(train_odd.shape, label_odd.shape, weight_odd.shape)
print(train_even.shape, label_even.shape, weight_even.shape)

# Load and configure the training feature data , labels and weights the DNN models
model_ = DNNModel(config,log_path)
model_.features_odd_data = train_odd
model_.features_even_data = train_even
model_.labels_odd_data = label_odd
model_.labels_even_data = label_even
model_.weights_odd_data = weight_odd
model_.weights_even_data = weight_even

# compile , train and save the model weights
if args.train:
    model_.compile_mclass_model()
    model_.train_mclass_model()
    model_.save_model_weights()

# predict DNN score model 
if args.predict:
    model_.pred_data_dict = dataset_.prepare_pred_data()
    model_.predict_mclass_model()



