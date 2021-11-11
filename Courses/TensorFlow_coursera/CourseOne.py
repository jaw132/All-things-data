# import libraries

import tensorflow as tf
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator



DESIRED_ACCURACY = 0.999

!wget --no-check-certificate \
  "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
   -O "/tmp/happy-or-sad.zip"

zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > DESIRED_ACCURACY:
            print("Accuracy = 100%, stop training")
            self.model.stop_training = True


callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(30, 30, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])



model.compile(optimizer=RMSprop(lr=0.001), loss="binary_crossentropy", metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/tmp/h-or-s', target_size=(30, 30),
    batch_size=20, class_mode='binary')

history = model.fit(train_generator,
                    steps_per_epoch=4,
                    epochs=10,
                    verbose=1)

# Extra code to visualise images
happy_dir = os.path.join('/tmp/h-or-s/happy')

sad_dir = os.path.join('/tmp/h-or-s/sad')

happy_files = os.listdir(happy_dir)
sad_files = os.listdir(sad_dir)

happypic = os.path.join(happy_dir, happy_files[0])
sadpic = os.path.join(sad_dir, sad_files[0])

img = mpimg.imread(sadpic)
plt.imshow(img)

plt.show()

