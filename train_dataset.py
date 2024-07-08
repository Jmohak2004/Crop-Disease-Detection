#Importing Dataset
#Dataset Link:- https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset


 # Importing all the libraries required for training
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Defining parameters
batch_size = 32
img_height = 128
img_width = 128

# Created the training dataset from the directory
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,  # Set a seed for reproducibility
    validation_split=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# Prefetched the dataset for performance optimization
training_set = training_set.prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])

# Applying data augmentation to the training dataset
training_set = training_set.map(lambda x, y: (data_augmentation(x, training=True), y))

# Prefetching the augmented dataset
training_set = training_set.prefetch(buffer_size=tf.data.AUTOTUNE)


# Defining parameters
batch_size = 32
img_height = 128
img_width = 128

# Created the validation dataset from the directory
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,  # Setting a seed for reproducibility
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# Prefetching the dataset for performance optimization
validation_set = validation_set.prefetch(buffer_size=tf.data.AUTOTUNE)

# Building CNN model for crop_disease_detection

import tensorflow as tf

# Building the Model
cnn = tf.keras.models.Sequential()

# Convolution and Pooling Layers
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128, 128, 3]))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Dropout(0.25))

# Flattening Layer
cnn.add(tf.keras.layers.Flatten())

# Fully Connected Layers
cnn.add(tf.keras.layers.Dense(units=1500, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.4))  # To avoid overfitting

# Output Layer
cnn.add(tf.keras.layers.Dense(units=38, activation='softmax'))

# Compile the model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
cnn.summary()

# Assuming training_set and validation_set are already created
history = cnn.fit(
    training_set,
    epochs=25,
    validation_data=validation_set
)

# Evaluate the model
results = cnn.evaluate(validation_set)
print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")

#Training set Accuracy
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)

#Validating set Accuracy
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)

#Saving the training Module
cnn.save('trained_plant_disease_model.keras')

#Accuracy Visulization using graph
epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='yellow',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='green',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()



