import os
import shutil
import datetime

import numpy as np
import pandas as pd
import pathlib

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Flatten, Softmax
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def load_data(train_path, val_path, batch_size):
    
    CLASS_LABELS = ['NORMAL', 'PNEUMONIA'] 

    def process_path(nb_class):
    
        def f(file_path):
            
            label = 0 if tf.strings.split(file_path, os.path.sep)[-2]=='NORMAL' else 1
            
            image = tf.io.read_file(file_path)    
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
         
            image = tf.image.resize(image, [127, 127], method='area')
            return image, label
    
        return f

    def reader_image(path_file, batch_size, nb_class):

        list_ds = tf.data.Dataset.list_files(path_file)
        labeled_ds = list_ds.map(process_path(nb_class))
    
        return labeled_ds.shuffle(100).batch(batch_size).prefetch(1)
    
    train_ds = reader_image(train_path, batch_size, 2)
    val_ds = reader_image(val_path, batch_size, 2)

    print(type(train_ds))


    for image, label in train_ds.take(1):
        df = pd.DataFrame(image[0, :, :, 0].numpy())
    
    print(f'Outoupt : \n image shape: {df.shape}')
    
    return train_ds, val_ds

def train_and_evaluate(args):  

    num_classes = 2
    cnn_model = tf.keras.Sequential([
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    optm = Adam(lr=0.0001)
    cnn_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
                      optimizer=optm, 
                      metrics=['accuracy'])

    checkpoint_path = os.path.join(args["output_dir"], "checkpoints/pneumonia")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True)
    
    train_ds, val_ds = load_data(args["train_data_path"], args["eval_data_path"], args["batch_size"])
    
    model_history = cnn_model.fit(
              train_ds,
              validation_data=val_ds,
              epochs=args["num_epochs"])
    
    EXPORT_PATH = os.path.join(
        args["output_dir"], datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    tf.saved_model.save(
        obj=cnn_model, export_dir=EXPORT_PATH)
    
    print("Exported trained model to {}".format(EXPORT_PATH))   
