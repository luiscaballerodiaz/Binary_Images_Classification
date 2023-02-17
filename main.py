import os
import utils
from keras import models
from keras import callbacks
from keras.preprocessing import image
from data_visualization import DataPlot
from data_split import DataSplit

# MODEL ACTION SETTING
# 0 --> Load models to evaluate test acc independently, then ensembling them, optimize weights and check test acc again
# 1 --> Run fold split cross validation using the training set
# 2 --> Run model using full training set
model_action = 0

# SETTINGS
avoid_overfit = True
batch_norm_flag = True
image_width = 150
image_height = 150
folds = 4
test_size = 0.25
learning_rate = 0.001
batch_size = 20
epochs = 250
l2 = 0.05
dropout = 0.5
patience_stop = 50
models_to_load = ['Fold1 - Trained model data augmentation and batch normalization.h5',
                  'Fold2 - Trained model data augmentation and batch normalization.h5',
                  'Fold3 - Trained model data augmentation and batch normalization.h5',
                  'Fold4 - Trained model data augmentation and batch normalization.h5']

base_dir = os.path.join(os.getcwd(), 'Images')
split = DataSplit(base_dir, folds, test_size)
visualization = DataPlot()
if avoid_overfit:
    description = 'data augmentation'
    l2_reg = l2
else:
    description = 'no data augmentation'
    l2_reg = 0
if batch_norm_flag:
    description += ' and batch normalization'
# Split images in train, validation and test sets and plot some original images
train_dirs, test_dirs, counts = split.create_working_folders()
train_dir = train_dirs[0]
train_cars_dir = train_dirs[1]
train_bikes_dir = train_dirs[2]
trainval_dir = train_dirs[3]
test_dir = test_dirs[0]
test_cars_dir = test_dirs[1]
test_bikes_dir = test_dirs[2]
testval_dir = test_dirs[3]
trainval_count = counts[0]
testval_count = counts[1]
train_count = counts[2]
test_count = counts[3]
car_images = visualization.plot_images(train_dir, 'CARS', image_width, image_height)
bike_images = visualization.plot_images(train_dir, 'BIKES', image_width, image_height)
# Create early stop callback, image generator for each set and visualize the effect of the applied data augmentation
data_augmentation = image.ImageDataGenerator(rescale=1. / 255, rotation_range=30, width_shift_range=0.2,
                                             height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                             fill_mode='nearest')
no_data_augmentation = image.ImageDataGenerator(rescale=1. / 255)
callbacks_list = callbacks.EarlyStopping(monitor='acc', patience=patience_stop)
visualization.data_augmentation(data_augmentation, car_images[1, :, :, :], 'CARS data augmentation')
visualization.data_augmentation(data_augmentation, bike_images[1, :, :, :], 'BIKES data augmentation')

if model_action == 0:
    # Load all models in models_to_load and calculate test accuracy
    models_list = [models.load_model(model) for model in models_to_load]
    test_generator = no_data_augmentation.flow_from_directory(test_dir, target_size=(image_width, image_height),
                                                              batch_size=batch_size, class_mode='binary',
                                                              interpolation='bilinear')
    results = [model.evaluate(test_generator, batch_size=batch_size) for model in models_list]
    ind = 0
    maxim = 0
    for i in range(len(results)):
        if results[i][1] > maxim:
            maxim = results[i][1]
            ind = i

    model = models_list[ind]
    # Visualize model outputs
    visualization.plot_activation_convnet(model, bike_images[1, :, :, :], 'Bike')
    visualization.plot_activation_convnet(model, car_images[1, :, :, :], 'Car')
    # Ensemble all models and check test accuracy
    ensembled_acc = utils.ensemble_models(models_to_load, train_bikes_dir, train_cars_dir, test_bikes_dir, test_cars_dir)
    print('TEST ACCURACY FOR THE ENSEMBLED MODEL: {}'.format(ensembled_acc))
    for i in range(len(results)):
        print('TEST LOSS AND TEST ACCURACY {}: {} and {}\n'.format(models_to_load[i], round(results[i][0], 4),
                                                                   round(results[i][1], 4)))

elif model_action == 1:
    # Create neural network model based on sequential layers and training it applying fold cross validation
    testval_loss = [''] * folds
    testval_acc = [''] * folds
    trainval_loss = [''] * folds
    trainval_acc = [''] * folds
    for i in range(folds):
        model = utils.create_cnn(image_width, image_height, learning_rate, avoid_overfit, batch_norm_flag,
                                 l2_reg, dropout)
        if avoid_overfit:
            trainval_generator = data_augmentation.flow_from_directory(trainval_dir[i],
                                                                       target_size=(image_width, image_height),
                                                                       batch_size=batch_size, class_mode='binary',
                                                                       interpolation='bilinear')
        else:
            trainval_generator = no_data_augmentation.flow_from_directory(trainval_dir[i],
                                                                          target_size=(image_width, image_height),
                                                                          batch_size=batch_size, class_mode='binary',
                                                                          interpolation='bilinear')
        testval_generator = no_data_augmentation.flow_from_directory(testval_dir[i],
                                                                     target_size=(image_width, image_height),
                                                                     batch_size=batch_size, class_mode='binary',
                                                                     interpolation='bilinear')
        history = model.fit(trainval_generator, steps_per_epoch=round(trainval_count / batch_size), epochs=epochs,
                            callbacks=callbacks_list, validation_data=testval_generator,
                            validation_steps=round(testval_count / batch_size))
        trainval_acc[i] = history.history['acc']
        trainval_loss[i] = history.history['loss']
        testval_acc[i] = history.history['val_acc']
        testval_loss[i] = history.history['val_loss']
        name = 'Fold' + str(i + 1) + ' - Trained model ' + description + '.h5'
        model.save(name)
    visualization.plot_results(trainval_acc, testval=testval_acc, description=description, folds=folds, tag='accuracy')
    visualization.plot_results(trainval_loss, testval=testval_loss, description=description, folds=folds, tag='loss')

elif model_action == 2:
    # Create neural network model based on sequential layers and training it against full training set
    model = utils.create_cnn(image_width, image_height, learning_rate, avoid_overfit, batch_norm_flag,
                             l2_reg, dropout)
    if avoid_overfit:
        train_generator = data_augmentation.flow_from_directory(train_dir, target_size=(image_width, image_height),
                                                                batch_size=batch_size, class_mode='binary',
                                                                interpolation='bilinear')
    else:
        train_generator = no_data_augmentation.flow_from_directory(train_dir, target_size=(image_width, image_height),
                                                                   batch_size=batch_size, class_mode='binary',
                                                                   interpolation='bilinear')
    history = model.fit(train_generator, steps_per_epoch=round(train_count / batch_size), epochs=epochs,
                        callbacks=callbacks_list)
    trainval_acc = history.history['acc']
    trainval_loss = history.history['loss']
    name = 'Trained model ' + description + '.h5'
    model.save(name)
    visualization.plot_results(trainval_acc, description, 'accuracy')
    visualization.plot_results(trainval_loss, description, 'loss')
