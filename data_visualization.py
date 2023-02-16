import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
            im = cv2.resize(im, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            im_array[i, :, :, :] = im
            ax[i].imshow(im)
        fig.suptitle('EXAMPLE FOR ' + folder + ' IMAGES', fontweight='bold', fontsize=24)
        fig.tight_layout()
        plt.savefig('Example for ' + folder + ' images.png', bbox_inches='tight')
        plt.close()
        return im_array

    def data_augmentation(self, image_generator, image, description):
        """Apply data augmentation to the input image and plot the first generated images"""
        im = image.reshape((1,) + image.shape)  # reshape to be (1, height, width, channels)
        fig, axes = plt.subplots(self.images_row, self.images_column, figsize=(self.fig_width, self.fig_height))
        ax = axes.ravel()
        for i, batch in zip(range(self.images_row * self.images_column), image_generator.flow(im, batch_size=1)):
            ax[i].imshow(batch[0, :, :, :])
        fig.suptitle('EXAMPLE FOR ' + description, fontweight='bold', fontsize=24)
        fig.tight_layout()
        plt.savefig('Example for ' + description + '.png', bbox_inches='tight')
        plt.close()

    def plot_results(self, trainval, description, tag, testval=0, folds=0):
        """Plot the accuracy results for train and validation sets per each epoch"""
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        fig, axes = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        if folds == 0:
            plt.plot(range(1, len(trainval) + 1), trainval, ls='-', lw=2, color='blue', label='Full Training ' + tag)
            text_str = '\nFULL TRAINING ' + tag.upper() + ' = ' + str(round(trainval[-1], 4))
        else:
            for i in range(len(trainval)):
                mess = 'split ' + str(i + 1)
                plt.plot(range(1, len(trainval[i]) + 1), trainval[i], ls='-', lw=2, color=colors[i % len(colors)],
                         label='Training ' + mess + ' ' + tag)
                plt.plot(range(1, len(testval[i]) + 1), testval[i],  ls='--', lw=2, color=colors[i % len(colors)],
                         label='Validation ' + mess + ' ' + tag)
            text_str = ''
            sum_trainval = 0
            sum_testval = 0
            for i in range(len(trainval)):
                if i < folds:
                    sum_trainval += trainval[i][-1]
                    sum_testval += testval[i][-1]
                    text_str += '\nSPLIT ' + str(i + 1) + ': TRAINING ' + tag.upper() + ' = ' + \
                                str(round(trainval[i][-1], 4)) + ' and VALIDATION ' + tag.upper() + ' = ' + \
                                str(round(testval[i][-1], 4))
            text_str += '\nAVERAGE SPLIT TRAINING ' + tag.upper() + ' = ' + str(round(sum_trainval / len(trainval), 4))
            text_str += '\nAVERAGE SPLIT VALIDATION ' + tag.upper() + ' = ' + str(round(sum_testval / len(testval), 4))
        plt.title(tag.upper() + ' RESULTS ' + description.upper() + text_str, fontweight='bold', fontsize=20)
        plt.legend()
        plt.grid()
        fig.tight_layout()
        plt.savefig(tag.title() + ' validation results ' + description + '.png', bbox_inches='tight')
        plt.close()

    def plot_activation_convnet(self, model, image, label):
        """Create a multioutput model using functional API having an output per each layer of the input model
        and plot the activation output of that model when predicting the input image"""
        layer_names = [layer.name for layer in model.layers]
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)  # Model with one output per layer
        im = image[:, :, :3].reshape((1,) + image[:, :, :3].shape)
        activations = activation_model.predict(im)
        layer_number = 1
        for layer, activation in zip(layer_names, activations):
            if 'conv2d' in layer.lower():
                layer = 'conv2d layer ' + str(layer_number)
                layer_number += 1
                n_features = activation.shape[3]
                n_rows = round(n_features / self.activation_row)
                fig, axes = plt.subplots(n_rows, self.activation_row, figsize=(self.fig_width, self.fig_height))
                ax = axes.ravel()
                for feat in range(n_features):
                    ax[feat].imshow(activation[0, :, :, feat])
                fig.suptitle('ACTIVATION ASSESSMENT ' + label.upper() + ' IMAGE WITH ' + layer.upper(),
                             fontweight='bold', fontsize=24)
                fig.tight_layout()
                plt.savefig('Activation assessment ' + label.upper() + ' image with ' + layer.upper() + '.png',
                            bbox_inches='tight')
                plt.close()
