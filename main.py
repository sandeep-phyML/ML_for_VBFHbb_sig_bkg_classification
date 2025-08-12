#this is the main code script
import os
from utils import Plot , PrepareDataset , DNNModel , DNNModelTraining , BasicMethods
import yaml

import keras
import tensorflow as tf
import math
# read the config file
basic_tool = BasicMethods()
config_file_name = "input_config.yml"
config = basic_tool.read_config_file(config_file_name)
log_path = os.path.join(config['output_log']['folder_path'],config['output_log']['file_name'])
basic_tool.create_log_folder(log_path)

# prepare the dataset for training
dataset_ = PrepareDataset( train_feature = config['train_features'] , weight_feature = config["weight_features"],output_log_path = log_path , nsample =config["nsamples"], is_resampling=config['is_resampling'],mass_norm = config['mass_norm'])
train_odd, train_even , weight_odd , weight_even , label_odd , label_even , mass_odd , mass_even = dataset_.get_np_feaweilabel_odd_even_train(folder_path= config["train_files_labels"]["folder_path"], file_label_list= config["train_files_labels"]["file_names_labels"])
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
model_.compile_mclass_model()
model_.train_mclass_model()
model_.save_model_weights()

# prepare the dataset for training get_np_feaweilabel_odd_even_train(self,folder_path:str, file_label_list: List[Dict[str, int]])

'''
# Train each model
odd_dataset = tf.data.Dataset.from_tensor_slices((
            train_odd,
            label_odd,
            weight_odd
        )).shuffle(1000).batch(config['batch_size'])

even_dataset = tf.data.Dataset.from_tensor_slices((
    train_even,
    label_even,
    weight_even
)).batch(config['batch_size'])

odd_dataset = odd_dataset.map(lambda x, y, w: ((x), y, w))
even_dataset = even_dataset.map(lambda x, y, w: ((x), y, w))

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', verbose=1, patience=5,
    restore_best_weights=True, start_from_epoch=10
)
def scheduler( epoch, lr, decay_rate=0.05, start_decay=10):
        if epoch < start_decay:
            return lr
        else:
            return lr * math.exp(-decay_rate * (epoch - start_decay))
lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

history_odd = odd_model_classifier.fit(
    odd_dataset,
    validation_data=even_dataset,
    epochs=config['num_epochs'],
    callbacks=[lr_scheduler, early_stopping]
)

history_even = even_model_classifier.fit(
    even_dataset,
    validation_data=odd_dataset,
    epochs=config['num_epochs'],
    callbacks=[lr_scheduler, early_stopping]
)
# Save the models

'''
