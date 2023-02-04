import os
import utils
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image

train_dir, test_dir, validation_dir = utils.create_working_folders(base_dir=os.path.join(os.getcwd(), 'Images'),
                                                                   validation_size=0.2, test_size=0.2)

train_cars_dir = os.path.join(train_dir, 'Cars')
fnames = [os.path.join(train_cars_dir, fname) for fname in os.listdir(train_cars_dir)]
im = plt.imread(fnames[3])
plt.imshow(im)
plt.show()

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

train_datagen = image.ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                         height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                         horizontal_flip=True, fill_mode='nearest')
test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20,
                                                        class_mode='binary')

history = model.fit_generator(train_generator, steps_per_epoch=60, epochs=40,
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
plt.show()
