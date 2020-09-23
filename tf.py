from tensorflow import keras
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bs = 10
num_epochs = 2

n_training = "Number of training images"
n_validation = "Number of validation images"

train_path = r'path to training images folder'
valid_path = r'path to validation images folder'

input_shape = (376, 672, 3)

aug = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

train_batches = aug.flow_from_directory(train_path, target_size=(376, 672), batch_size=bs, class_mode='sparse')
valid_batches = aug.flow_from_directory(valid_path, target_size=(376, 672), batch_size=bs, class_mode='sparse')

print(train_batches.class_indices)
print(valid_batches.class_indices)

model = keras.Sequential([
    keras.layers.Conv2D(32, (5, 5), strides=(2, 2), input_shape=input_shape),
    keras.layers.BatchNormalization(axis=3),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(32, (3, 3), strides=(2, 2)),
    keras.layers.BatchNormalization(axis=3),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), strides=(2, 2)),
    keras.layers.BatchNormalization(axis=3),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, (3, 3), strides=(2, 2)),
    keras.layers.BatchNormalization(axis=3),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), strides=(1, 1)),
    keras.layers.BatchNormalization(axis=3),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(128, (3, 3), strides=(1, 1)),
    keras.layers.BatchNormalization(axis=3),
    keras.layers.Activation('relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(keras.optimizers.SGD(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=n_training//bs, validation_data=valid_batches,
                    validation_steps=n_validation//bs, epochs=num_epochs)
model.save("trained_model4.h5")

