import os
import utils
from keras import layers
from keras import models
from keras import optimizers
from keras import callbacks
from keras.preprocessing import image
from data_visualization import DataPlot


image_width = 150
image_height = 150
run_model = True
description = 'data augmentation'

name = 'Trained model ' + description + '.h5'
visualization = DataPlot()
train_dir, test_dir, validation_dir = utils.create_working_folders(base_dir=os.path.join(os.getcwd(), 'Images'),
                                                                   validation_size=0.2, test_size=0.2)
car_images = visualization.plot_images(train_dir, 'Cars', image_width, image_height)
bike_images = visualization.plot_images(train_dir, 'Bikes', image_width, image_height)

train_datagen = image.ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                         height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                         horizontal_flip=True, fill_mode='nearest')
test_datagen = image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(image_width, image_height),
                                                    batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(image_width, image_height),
                                                        batch_size=20, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(image_width, image_height),
                                                  batch_size=20, class_mode='binary')

visualization.data_augmentation(train_datagen, car_images[1, :, :, :], 'Cars')
visualization.data_augmentation(train_datagen, bike_images[1, :, :, :], 'Bikes')

if run_model:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
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
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])
    callbacks_list = callbacks.EarlyStopping(monitor='acc', patience=2)
    history = model.fit(train_generator, steps_per_epoch=120, epochs=30, callbacks=callbacks_list,
                        validation_data=validation_generator, validation_steps=40)
    results = model.evaluate(test_generator, batch_size=20)
    visualization.plot_results(history, description, round(results[1], 2))
    model.save(name)
else:
    model = models.load_model(name)

visualization.plot_activation_convnet(model, bike_images[1, :, :, :])
results = model.evaluate(test_generator, batch_size=20)
print('TEST ACCURACY: {}'.format(results[1]))
