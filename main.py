import os
import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import sys


train_dir, test_dir, validation_dir = utils.create_working_folders(base_dir=os.path.join(os.getcwd(), 'Images'),
                                                                   validation_size=0.2, test_size=0.2)

image_width = 150
image_height = 150
fig_width = 20
fig_height = 10
images_row = 3
images_column = 3

car_im_array = np.zeros([images_row * images_column, image_height, image_width, 4])
image_dir = os.path.join(train_dir, 'Cars')
fnames = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
fig, axes = plt.subplots(images_row, images_column, figsize=(fig_width, fig_height))
ax = axes.ravel()
for i in range(images_row * images_column):
    im = cv2.imread(fnames[i], cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    im = cv2.resize(im, (image_width, image_height), interpolation=cv2.INTER_AREA)
    car_im_array[i, :, :, :] = im
    ax[i].imshow(im)
fig.suptitle('EXAMPLE FOR CAR IMAGES', fontweight='bold', fontsize=24)
fig.tight_layout()
plt.savefig('Example for CAR images.png', bbox_inches='tight')
plt.clf()

bike_im_array = np.zeros([images_row * images_column, image_height, image_width, 4])
image_dir = os.path.join(train_dir, 'Bikes')
fnames = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
fig, axes = plt.subplots(images_row, images_column, figsize=(fig_width, fig_height))
ax = axes.ravel()
for i in range(images_row * images_column):
    im = cv2.imread(fnames[i], cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    im = cv2.resize(im, (image_width, image_height), interpolation=cv2.INTER_AREA)
    bike_im_array[i, :, :, :] = im
    ax[i].imshow(im)
fig.suptitle('EXAMPLE FOR BIKE IMAGES', fontweight='bold', fontsize=24)
fig.tight_layout()
plt.savefig('Example for BIKE images.png', bbox_inches='tight')
plt.clf()

train_datagen = image.ImageDataGenerator(rescale=1./255, rotation_range=90, width_shift_range=0.2,
                                         height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                         horizontal_flip=True, fill_mode='nearest')

im = bike_im_array[1, :, :, :]
im = im.reshape((1,) + im.shape)  # reshape to be (1, height, width, channels)
fig, axes = plt.subplots(images_row, images_column, figsize=(fig_width, fig_height))
ax = axes.ravel()
for i, batch in zip(range(images_row * images_column), train_datagen.flow(im, batch_size=1)):
    ax[i].imshow(batch[0, :, :, :])
fig.suptitle('EXAMPLE FOR DATA AUGMENTATION WITH A BIKE IMAGE', fontweight='bold', fontsize=24)
fig.tight_layout()
plt.savefig('Example for BIKE data augmentation.png', bbox_inches='tight')
plt.clf()

im = car_im_array[1, :, :, :]
im = im.reshape((1,) + im.shape)  # reshape to be (1, height, width, channels)
fig, axes = plt.subplots(images_row, images_column, figsize=(fig_width, fig_height))
ax = axes.ravel()
for i, batch in zip(range(images_row * images_column), train_datagen.flow(im, batch_size=1)):
    ax[i].imshow(batch[0, :, :, :])
fig.suptitle('EXAMPLE FOR DATA AUGMENTATION WITH A CAR IMAGE', fontweight='bold', fontsize=24)
fig.tight_layout()
plt.savefig('Example for CAR data augmentation.png', bbox_inches='tight')
plt.clf()

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20,
                                                    class_mode='binary')

test_datagen = image.ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20,
                                                        class_mode='binary')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=60, epochs=2,
                              validation_data=validation_generator, validation_steps=40)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.grid()
plt.show()
