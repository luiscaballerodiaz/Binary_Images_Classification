import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import models


class DataPlot:
    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10
        self.images_row = 3
        self.images_column = 3
        self.activation_row = 16

    def plot_images(self, directory, folder, image_width, image_height):
        """Plot the first images of the input folder found in the input directory sized accordingly to the input
        image width and height"""
        im_array = np.zeros([self.images_row * self.images_column, image_height, image_width, 4])
        image_dir = os.path.join(directory, folder)
        fnames = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        fig, axes = plt.subplots(self.images_row, self.images_column, figsize=(self.fig_width, self.fig_height))
        ax = axes.ravel()
        for i in range(self.images_row * self.images_column):
            im = cv2.imread(fnames[i], cv2.IMREAD_UNCHANGED)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
            im = cv2.resize(im, (image_width, image_height), interpolation=cv2.INTER_AREA)
            im_array[i, :, :, :] = im
            ax[i].imshow(im)
        fig.suptitle('EXAMPLE FOR ' + folder.upper() + ' IMAGES', fontweight='bold', fontsize=24)
        fig.tight_layout()
        plt.savefig('Example for ' + folder.upper() + ' images.png', bbox_inches='tight')
        plt.clf()
        return im_array

    def data_augmentation(self, image_generator, image, label=''):
        """Apply data augmentation to the input image and plot the first generated images"""
        im = image.reshape((1,) + image.shape)  # reshape to be (1, height, width, channels)
        fig, axes = plt.subplots(self.images_row, self.images_column, figsize=(self.fig_width, self.fig_height))
        ax = axes.ravel()
        for i, batch in zip(range(self.images_row * self.images_column), image_generator.flow(im, batch_size=1)):
            ax[i].imshow(batch[0, :, :, :])
        fig.suptitle('EXAMPLE FOR DATA AUGMENTATION WITH A ' + label.upper() + ' IMAGE', fontweight='bold', fontsize=24)
        fig.tight_layout()
        plt.savefig('Example for ' + label.upper() + ' data augmentation.png', bbox_inches='tight')
        plt.clf()

    def plot_results(self, history, description, test):
        """Plot the accuracy results for train and validation sets per each epoch"""
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        epochs = range(1, len(acc) + 1)
        fig, axes = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('ACCURACY RESULTS ' + description.upper() + ' WITH TEST ACCURACY = ' + str(test),
                  fontweight='bold', fontsize=24)
        plt.legend()
        plt.grid()
        fig.tight_layout()
        plt.savefig('Accuracy results ' + description.upper() + '.png', bbox_inches='tight')
        plt.clf()

    def plot_activation_convnet(self, model, image):
        """Create a multioutput model using functional API having an output per each layer of the input model
        and plot the activation output of that model when predicting the input image"""
        layer_names = [layer.name for layer in model.layers]
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)  # Model with one output per layer
        im = image[:, :, :3].reshape((1,) + image[:, :, :3].shape)
        activations = activation_model.predict(im)
        for layer, activation in zip(layer_names, activations):
            if 'conv2d' in layer.lower():
                n_features = activation.shape[3]
                n_rows = round(n_features / self.activation_row)
                fig, axes = plt.subplots(n_rows, self.activation_row, figsize=(self.fig_width, self.fig_height))
                ax = axes.ravel()
                for feat in range(n_features):
                    ax[feat].imshow(activation[0, :, :, feat])
                fig.suptitle('ACTIVATION ASSESSMENT FOR ' + layer.upper() + ' LAYER', fontweight='bold', fontsize=24)
                fig.tight_layout()
                plt.savefig('Activation assessment for ' + layer.upper() + ' layer.png', bbox_inches='tight')
                plt.clf()
