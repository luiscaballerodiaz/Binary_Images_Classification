import os
import cv2
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from scipy.optimize import minimize


def create_cnn(image_width, image_height, learning_rate, avoid_overfit, batch_norm_flag, l2_reg, dropout):
    model = models.Sequential()
    model.add(layers.SeparableConv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_reg),
                                     input_shape=(image_width, image_height, 3)))
    if batch_norm_flag:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SeparableConv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm_flag:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SeparableConv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm_flag:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SeparableConv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm_flag:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    if avoid_overfit:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=learning_rate),
                  metrics=['acc'])
    return model


def ensemble_models(models_input, train_dir_class0, train_dir_class1, test_dir_class0, test_dir_class1):
    models_list = [models.load_model(model) for model in models_input]
    model_input = models_list[0].input
    image_width = model_input.shape[-3]
    image_height = model_input.shape[-2]
    y_pred_train, y_true_train = make_predictions(models_list, train_dir_class0, train_dir_class1,
                                                  image_width, image_height)
    y_pred_test, y_true_test = make_predictions(models_list, test_dir_class0, test_dir_class1,
                                                image_width, image_height)
    test_acc = 100
    for i in range(10000):
        weights_ini = np.random.rand(len(models_list))
        weights_ini /= np.sum(weights_ini)
        opt_weights = minimize(fun=maximize_acc,
                               x0=weights_ini,
                               method='SLSQP',
                               args=(y_true_train, y_pred_train),
                               bounds=[(0, 1)] * y_pred_train.shape[1],
                               options={'disp': True, 'maxiter': 10000, 'eps': 1e-10, 'ftol': 1e-8},
                               constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
        acc = np.sum(np.absolute(y_true_train - np.round(np.dot(y_pred_train, opt_weights.x)))) / y_true_train.shape[0]
        if acc < test_acc:
            test_acc = acc
            weights_opt = opt_weights.x
    print('Optimal weights: {}'.format(weights_opt))
    test_acc = 1 - np.sum(np.absolute(y_true_test - np.round(np.dot(y_pred_test, weights_opt)))) / y_true_test.shape[0]
    return test_acc


def maximize_acc(weights, y_true, y_pred):
    """ Calculate the score of a weighted model predictions"""
    return np.sum(np.absolute(y_true - np.round(np.dot(y_pred, weights)))) / y_true.shape[0]


def make_predictions(models_list, dir_class0, dir_class1, image_width, image_height):
    fnames_class0 = [os.path.join(dir_class0, fname) for fname in os.listdir(dir_class0)]
    fnames_class1 = [os.path.join(dir_class1, fname) for fname in os.listdir(dir_class1)]
    im_array = np.zeros([len(fnames_class0) + len(fnames_class1), image_height, image_width, 3])
    y_pred = np.zeros([len(fnames_class0) + len(fnames_class1), len(models_list)])
    y_true = np.zeros([len(fnames_class0) + len(fnames_class1)])
    y_true[:len(fnames_class0)] = 0
    y_true[len(fnames_class0):] = 1
    for i in range(len(fnames_class0)):
        im = cv2.imread(fnames_class0[i], cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        im_array[i, :, :, :] = im / 255
    for i in range(len(fnames_class1)):
        im = cv2.imread(fnames_class1[i], cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        im_array[len(fnames_class0) + i, :, :, :] = im / 255
    for n, model in zip(range(len(models_list)), models_list):
        y_pred[:, n] = model.predict(im_array).squeeze()
    return y_pred, y_true
