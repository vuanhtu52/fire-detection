"""
#################################
 Training phase after demonstration: This module uses Keras and Tensor flow to train the image classification problem
 for the labeling fire and non-fire data based on the aerial images.
 Training and Validation Data: Item 7 on https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
 Keras version: 2.4.0
 Tensorflow Version: 2.3.0
 GPU: Nvidia RTX 2080 Ti
 OS: Ubuntu 18.04
#################################
"""

#########################################################
# import libraries

import os.path
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.callbacks import Callback
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models
from pathlib import Path
import time
import json

from config import new_size
from plotdata import plot_training
from config import Config_classification

#########################################################
# Global parameters and definition

data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

image_size = (new_size.get('width'), new_size.get('height'))
batch_size = Config_classification.get('batch_size')
save_model_flag = Config_classification.get('Save_Model')
epochs = Config_classification.get('Epochs')
architecture = Config_classification.get("architecture")
model_path = Config_classification.get("model_path")
learning_rate = Config_classification.get("learning_rate")

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.Accuracy(name='accuracy'),
    keras.metrics.BinaryAccuracy(name='bin_accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
]


#########################################################
# Function definition

def train_keras():
    """
    This function train a DNN model based on Keras and Tensorflow as a backend. At first, the directory of Fire and
    Non_Fire images should be defined for the model, then the model is defined, compiled and fitted over the training
    and validation set. At the end, the models is saved based on the *.h5 parameters and weights. Training accuracy and
    loss are demonstrated at the end of this function.
    :return: None, Save the trained model and plot accuracy and loss on train and validation dataset.
    """
    # This model is implemented based on the guide in Keras (Xception network)
    # https://keras.io/examples/vision/image_classification_from_scratch/
    
    print(" --------- Training --------- ")

    # Create the model's directory if not exist
    Path(model_path).mkdir(parents=True, exist_ok=True)

    dir_fire = 'frames/Training/Fire/'
    dir_no_fire = 'frames/Training/No_Fire/'
    dir_smoke = 'frames/Training/Smoke/'

    # Count images for each class: 0 is Fire and 1 is NO_Fire
    fire = len([name for name in os.listdir(dir_fire) if os.path.isfile(os.path.join(dir_fire, name))])
    no_fire = len([name for name in os.listdir(dir_no_fire) if os.path.isfile(os.path.join(dir_no_fire, name))])
    smoke = len([name for name in os.listdir(dir_smoke) if os.path.isfile(os.path.join(dir_smoke, name))])
    total = fire + no_fire + smoke

    weight_for_fire = (1 / fire) * total / 3.0
    weight_for_no_fire = (1 / no_fire) * total / 3.0
    weight_for_smoke = (1 / smoke) * total / 3.0

    print("Weight for class Fire : {:.2f}".format(weight_for_fire))
    print("Weight for class No_fire : {:.2f}".format(weight_for_no_fire))
    print("Weight for class Smoke : {:.2f}".format(weight_for_smoke))

    class_weights = {
        0: weight_for_fire,
        1: weight_for_no_fire,
        2: weight_for_smoke
    }

    # Prepare datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Training", 
        validation_split=0.2, 
        subset="training", 
        seed=1337, 
        image_size=image_size,
        batch_size=batch_size, 
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Training",
        validation_split=0.2, 
        subset="validation", 
        seed=1337, 
        image_size=image_size,
        batch_size=batch_size, 
        shuffle=True
    )

    # Prefetching for performance optimisation
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    # Define model
    if architecture == "xception":
        model = make_xception_model(input_shape=image_size + (3,), num_classes=3)
    elif architecture == "efficientnet":
        model = make_efficientnet_model(input_shape=image_size + (3,), num_classes=3)
    elif architecture == "efficientnet_pretrained":
        model = make_efficientnet_model_pretrained(input_shape=image_size + (3,), num_classes=3)
    elif architecture == "inceptionv3":
        model = make_inceptionv3_model(input_shape=image_size + (3,), num_classes=3)
    keras.utils.plot_model(model, show_shapes=True)

    # Add the custom logger callback
    custom_logger = CustomEpochLogger(model_path)

    # Start time
    start_time = time.time()

    # Save config information
    with open(f"{model_path}/config.json", "w") as f:
        json.dump(Config_classification, f, indent=4)

    # Model training
    callbacks = [keras.callbacks.ModelCheckpoint(f"{model_path}/checkpoints/" + "save_at_{epoch}.h5"), custom_logger]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    res_fire = model.fit(
        train_ds, 
        epochs=epochs, 
        callbacks=callbacks, 
        validation_data=val_ds, 
        batch_size=batch_size,
        class_weight=class_weights
    )

    # End time
    end_time = time.time()

    # Calculate total training time 
    training_time = end_time - start_time

    # Save training time
    with open(f"{model_path}/logs.txt", "a") as f:
        f.write(f"Total training time: {training_time} s")

    layers_len = len(model.layers)

    if save_model_flag:
        model.save(model_path)
    if Config_classification.get('TrainingPlot'):
        plot_training(res_fire, 'KerasModel', layers_len)


def make_xception_model(input_shape, num_classes):
    """
    This function define the DNN Model based on the Keras example.
    :param input_shape: The requested size of the image
    :param num_classes: In this classification problem, there are two classes: 1) Fire and 2) Non_Fire.
    :return: The built model is returned
    """
    print("Making xception model")

    inputs = keras.Input(shape=input_shape)
    # x = data_augmentation(inputs)  # 1) First option
    x = inputs  # 2) Second option

    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    # for size in [128, 256, 512, 728]:
    for size in [8]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)

        x = layers.add([x, residual])
        previous_block_activation = x
    x = layers.SeparableConv2D(8, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs, name="model_fire")


def make_efficientnet_model_pretrained(input_shape, num_classes):
    """
    This function defines the DNN Model using EfficientNetB0.
    :param input_shape: The requested size of the image (height, width, channels)
    :param num_classes: The number of classes to classify (e.g., 3 for Fire, No Fire, Smoke)
    :return: The built model is returned
    """
    inputs = keras.Input(shape=input_shape)

    # Data augmentation (optional)
    x = data_augmentation(inputs)  # Optional: If you want to apply data augmentation
    # x = inputs  # Use this if you don't want to apply data augmentation

    # Preprocessing - rescale pixel values (0-1 range)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)

    # Load the pre-trained EfficientNet model with weights from ImageNet, without the top layers
    base_model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the base model to prevent updating pre-trained weights during training
    base_model.trainable = False

    # Add a global pooling layer and a fully connected layer for classification
    x = layers.GlobalAveragePooling2D()(base_model.output)
    
    # Add a dropout layer to avoid overfitting
    x = layers.Dropout(0.5)(x)
    
    # Add the final classification layer depending on the number of classes
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    # Output layer
    outputs = layers.Dense(units, activation=activation)(x)

    # Create the full model
    model = models.Model(inputs, outputs, name="EfficientNet_fire")

    return model


def make_efficientnet_model(input_shape, num_classes):
    """
    Build an EfficientNet model from scratch (no pre-trained weights).
    
    :param input_shape: The shape of the input image (height, width, channels)
    :param num_classes: Number of classes for classification (e.g., 2 for binary classification)
    :return: The compiled EfficientNet model
    """
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # EfficientNet base model without pre-trained weights
    efficientnet_base = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)

    # Add custom layers on top of EfficientNet base
    x = data_augmentation(inputs)
    x = efficientnet_base(x)
    x = layers.GlobalAveragePooling2D()(x)  # Global average pooling to reduce the spatial dimensions
    x = layers.Dropout(0.5)(x)  # Dropout to prevent overfitting

    # Output layer (softmax for multi-class, sigmoid for binary classification)
    if num_classes == 2:
        output = layers.Dense(1, activation='sigmoid')(x)  # Binary classification
    else:
        output = layers.Dense(num_classes, activation='softmax')(x)  # Multi-class classification

    # Create the final model
    model = models.Model(inputs, output)

    return model


def make_inceptionv3_model(input_shape, num_classes):
    """
    This function builds the InceptionV3 model without using pretrained weights.
    
    :param input_shape: Tuple specifying the shape of the input (e.g., (224, 224, 3) for RGB images).
    :param num_classes: Number of output classes for the classification problem.
    :return: Keras model instance with InceptionV3 architecture.
    """

    print("Making inceptionv3 model")
    
    # Define the input
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)

    # Load InceptionV3 model without pre-trained weights and include the top fully-connected layers
    base_model = InceptionV3(include_top=False, weights=None, input_tensor=x)

    # Add a Global Average Pooling layer 
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Add a fully-connected layer with dropout for regularization
    # x = layers.Dropout(0.5)(x)

    # Output layer with softmax or sigmoid activation depending on number of classes
    if num_classes == 2:
        # Binary classification
        activation = "sigmoid"
        units = 1
    else:
        # Multi-class classification
        activation = "softmax"
        units = num_classes

    # Add the output layer
    outputs = layers.Dense(units, activation=activation)(x)

    # Build the model
    model = models.Model(inputs, outputs, name="InceptionV3")

    return model


# Custom callback to log training information after every epoch
class CustomEpochLogger(Callback):
    def __init__(self, model_path):
        super(CustomEpochLogger, self).__init__()
        self.log_file = f"{model_path}/logs.txt"

    def on_epoch_end(self, epoch, logs=None):
        """Logs the epoch number and associated metrics to a file after each epoch"""
        logs = logs or {}
        with open(self.log_file, "a") as f:
            log_message = f"Epoch {epoch + 1}:\n"
            for key, value in logs.items():
                log_message += f"  {key}: {value:.4f}\n"
            log_message += "\n"
            f.write(log_message)
        print(f"Epoch {epoch + 1} logs written to {self.log_file}")

