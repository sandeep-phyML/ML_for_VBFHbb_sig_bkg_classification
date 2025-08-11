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
# get variables for the DNN model 
input_shape = len(config['train_features'])
train_files = config['train_files_labels']['file_names_labels']
output_shape = 1
activation = config['act_output_layer_binary']
loss_function = config['loss_function_binary']
if len(train_files) > 2:
    output_shape = len(train_files)
    activation = config["act_output_layer_mclass"]
    loss_function = config['loss_function_mclass']


model_ = DNNModel(input_shape=input_shape, nclass=output_shape, activation=activation,learning_rate=config['learning_rate'],loss_function = loss_function )
odd_model_classifier , even_model_classifier = model_.odd_classifier , model_.even_classifier
model_.compile_mclass_model(odd_model_classifier, even_model_classifier)


# dnn_training = DNNModelTraining(learning_rate= config['learning_rate'],loss_function = loss_function )
# dnn_training.compile_mclass_model(odd_model_classifier, even_model_classifier)


print("odd_model_classifier compiled? :", model_.is_compiled(odd_model_classifier))
print("even_model_classifier compiled? :", model_.is_compiled(even_model_classifier))

# odd_model_classifier.compile(
#     optimizer=optimizer_odd,
#     loss=loss_function,
#     metrics=[
#         'accuracy'
#     ]
# )
'''

dnn_model = Dnn(shape ,mclass_activation , 5 )
model = dnn_model.model4
model.compile(
    optimizer=optimizer,
    loss='SparseCategoricalCrossentropy',  # or keras.losses.BinaryCrossentropy()
    metrics=[
        'accuracy'
    ]
)

'''
# even_model_classifier.compile(
#     optimizer=optimizer_even,
#     loss=loss_function,
#     metrics=[
#         'accuracy'
#     ]
# )

odd_model_classifier.summary()  
even_model_classifier.summary()
'''
# prepare the dataset for training get_np_feaweilabel_odd_even_train(self,folder_path:str, file_label_list: List[Dict[str, int]])
dataset_ = PrepareDataset( train_feature = config['train_features'] , weight_feature = config["weight_features"],output_log_path = log_path , nsample =config["nsamples"], is_resampling=config['is_resampling'],mass_norm = config['mass_norm'])

train_odd, train_even , weight_odd , weight_even , label_odd , label_even , mass_odd , mass_even = dataset_.get_np_feaweilabel_odd_even_train(folder_path= config["train_files_labels"]["folder_path"], file_label_list= config["train_files_labels"]["file_names_labels"])
print(train_odd.shape, label_odd.shape, weight_odd.shape)
print(train_even.shape, label_even.shape, weight_even.shape)

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
