#this is the main code script
import os
from utils import Plot , PrepareDataset , DNNModel
import yaml
# read the config file
config_file_name = "input_config.yml"
with open(config_file_name, 'r') as file:
    config = yaml.safe_load(file)
log_path = os.path.join(config['output_log']['folder_path'],config['output_log']['file_name'])
with open(log_path, 'w') as log_file:
    log_file.write(f"Log file created at: {log_path}\n")
# get variables for the DNN model 
input_shape = len(config['train_features'])
train_files = config['train_files_labels']['file_names_labels']
output_shape = 1
activation = config['act_output_layer_binary']
if len(train_files) > 2:
    output_shape = len(train_files)
    activation = config["act_output_layer_mclass"]

# create the two model object input_shape: int, activation: str = 'sigmoid', nclass: int = 1
odd_model = DNNModel(input_shape = input_shape, nclass = output_shape, activation = activation)
odd_model_classifier = odd_model.classifier
even_model = DNNModel(input_shape = input_shape, nclass = output_shape, activation = activation)
even_model_classifier = even_model.classifier
odd_model_classifier.summary()  
even_model_classifier.summary()

# prepare the dataset for training get_np_feaweilabel_odd_even_train(self,folder_path:str, file_label_list: List[Dict[str, int]])
dataset_ = PrepareDataset( train_feature = config['train_features'] , weight_feature = config["weight_features"],output_log_path = log_path , nsample =config["nsamples"], is_resampling=config['is_resampling'],mass_norm = config['mass_norm'])

train_odd, train_even , weight_odd , weight_even , label_odd , label_even , mass_odd , mass_even = dataset_.get_np_feaweilabel_odd_even_train(folder_path= config["train_files_labels"]["folder_path"], file_label_list= config["train_files_labels"]["file_names_labels"])