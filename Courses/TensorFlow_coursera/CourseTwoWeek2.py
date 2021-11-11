#import libraries

import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt

#!wget --no-check-certificate \
 #   "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
  #  -O "/tmp/cats-and-dogs.zip"

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))

try:
    home_dir = '/tmp'
    catvdog_dir = home_dir + '/cats-v-dogs'
    train_dir, test_dir = catvdog_dir + '/training', catvdog_dir + '/testing'
    os.mkdir(catvdog_dir)
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    for _dir in [train_dir, test_dir]:
        dog_dir = _dir + '/dogs'
        cat_dir = _dir + '/cats'
        os.mkdir(dog_dir)
        os.mkdir(cat_dir)

except OSError:
    pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    for file in os.listdir(TRAINING):
        os.remove(TRAINING + file)

    for file in os.listdir(TESTING):
        os.remove(TESTING + file)

    source_files = os.listdir(SOURCE)
    for file in source_files:
        target_file = SOURCE + file
        if os.path.getsize(target_file) == 0:
            source_files.remove(file)
            print(file + " is zero length, so ignoring")

    random.sample(source_files, len(source_files))
    split_index = int(SPLIT_SIZE * len(source_files))
    train_files, test_files = source_files[:split_index], source_files[split_index:]
    for file in train_files:
        target_file = SOURCE + file
        copyfile(target_file, TRAINING + file)

    for file in test_files:
        target_file = SOURCE + file
        copyfile(target_file, TESTING + file)


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

TRAINING_DIR = "/tmp/cats-v-dogs/training"
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
TRAINING_DIR, target_size=(150, 150), batch_size=25, class_mode='binary')

VALIDATION_DIR = "/tmp/cats-v-dogs/testing"
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
VALIDATION_DIR, target_size=(150, 150), batch_size=25, class_mode='binary')

history = model.fit(train_generator,
                    epochs=15,
                    steps_per_epoch=900,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=100)




#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')



