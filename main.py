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
folds = 5
batch_size = 20
epochs = 100
patience = 10
description = 'data augmentation'

visualization = DataPlot()
name = 'Trained model ' + description + '.h5'
# Split images in train, validation and test sets and plot some original images
train_dir, test_dir, trainval_dir, testval_dir, trainval_count, testval_count = \
    utils.create_working_folders(base_dir=os.path.join(os.getcwd(), 'Images'), folds=folds, test_size=0.2)
car_images = visualization.plot_images(train_dir, 'Cars', image_width, image_height)
bike_images = visualization.plot_images(train_dir, 'Bikes', image_width, image_height)
# Create image generator for each set and visualize the effect of the applied data augmentation
train_datagen = image.ImageDataGenerator(rescale=1. / 255, rotation_range=30, width_shift_range=0.2,
                                         height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                         horizontal_flip=True, fill_mode='nearest')
test_datagen = image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(image_width, image_height),
                                                    batch_size=batch_size, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(image_width, image_height),
                                                  batch_size=batch_size, class_mode='binary')
visualization.data_augmentation(train_datagen, car_images[1, :, :, :], 'Cars with ' + description)
visualization.data_augmentation(train_datagen, bike_images[1, :, :, :], 'Bikes with ' + description)

if run_model:
    # Create the neural network based on sequential layers
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
    callbacks_list = callbacks.EarlyStopping(monitor='acc', patience=patience)
    trainval_accuracy = [''] * folds
    testval_accuracy = [''] * folds
    test_accuracy = [''] * folds
    for i in range(len(trainval_dir)):
        train_generator = test_datagen.flow_from_directory(trainval_dir[i], target_size=(image_width, image_height),
                                                           batch_size=batch_size, class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(testval_dir[i], target_size=(image_width, image_height),
                                                                batch_size=batch_size, class_mode='binary')
        history = model.fit(train_generator, steps_per_epoch=round(trainval_count / batch_size), epochs=epochs,
                            callbacks=callbacks_list, validation_data=validation_generator,
                            validation_steps=round(testval_count / batch_size))
        trainval_accuracy[i] = history.history['acc']
        testval_accuracy[i] = history.history['val_acc']
        test_accuracy[i] = model.evaluate(test_generator, batch_size=batch_size)[1]
        model.save('Fold' + str(i + 1) + ' - ' + name)
    visualization.plot_results(trainval_accuracy, testval_accuracy, test_accuracy, description)
else:
    model = models.load_model(name)

visualization.plot_activation_convnet(model, bike_images[1, :, :, :], 'Bike')
visualization.plot_activation_convnet(model, car_images[1, :, :, :], 'Car')
results = model.evaluate(test_generator, batch_size=batch_size)
print('TEST ACCURACY: {}\n'.format(results[1]))
