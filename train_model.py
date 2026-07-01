import os
import json
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def setup_kaggle():
    print("Setting up Kaggle credentials...")
    kaggle_creds = {
        "username": "remixeser",
        "key": "KGAT_f1197d8dc669305216daa47faa0d7d44"
    }
    
    home_dir = os.path.expanduser('~')
    kaggle_dir = os.path.join(home_dir, '.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    creds_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(creds_path, 'w') as f:
        json.dump(kaggle_creds, f)
        
    try:
        os.chmod(creds_path, 0o600)
    except Exception:
        pass # Ignore permission error on Windows if any

def download_dataset():
    print("Downloading coffee bean dataset from Kaggle...")
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle CLI...")
        os.system("pip install -q kaggle")
        import kaggle
        
    kaggle.api.dataset_download_files('gpiosenka/coffee-bean-dataset-resized-224-x-224', path='.', unzip=True)
    print("Dataset downloaded and extracted.")

def train():
    setup_kaggle()
    if not os.path.exists('train') or not os.path.exists('test'):
        download_dataset()
        
    data_dir = '.'
        
    print("Loading metadata...")
    df = pd.read_csv(os.path.join(data_dir, 'Coffee Bean.csv'))
    
    # Correct filepaths to absolute or local relative
    df['filepaths'] = df['filepaths'].apply(lambda x: os.path.join(data_dir, x))
    
    train_df = df[df['data set'] == 'train'].reset_index(drop=True)
    test_df = df[df['data set'] == 'test'].reset_index(drop=True)
    
    # 90-10 train-validation split
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.1, 
        random_state=42, 
        stratify=train_df['labels']
    )
    
    IMG_SIZE = 128
    
    print("Setting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepaths',
        y_col='labels',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='sparse',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepaths',
        y_col='labels',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='sparse',
        shuffle=False
    )
    
    print("Building model...")
    l2_strength = 0.0001
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), kernel_regularizer=l2(l2_strength)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(l2_strength)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(l2_strength)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, kernel_regularizer=l2(l2_strength)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Dense(128, kernel_regularizer=l2(l2_strength)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Dense(4))
    model.add(Activation("softmax"))
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        metrics=["accuracy"]
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=0.00001)
    
    print("Starting training (epochs=3 for quick validation. Increase epochs as needed)...")
    model.fit(
        train_generator,
        epochs=10, # default 10 epochs for demo. User can adjust.
        validation_data=val_generator,
        callbacks=[early_stopping, lr_reduction]
    )
    
    save_path = './backend/models/FineTunedCoffeeBeanCNN.keras'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved successfully to {save_path}")

if __name__ == "__main__":
    train()
