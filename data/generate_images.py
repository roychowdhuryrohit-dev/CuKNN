import numpy as np
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        


input_path = '/WAVE/users2/unix/klor/ycho_lab/kimsong_lor/csen_319/CuKNN/data/ubyte_data' # Change this your path
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


def save_images_as_png(images, labels, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Loop over each image and label, save them into respective directories
    for i, (img, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(save_dir, str(label))  # Create label-based subdirectories
        if not os.path.exists(label_dir):
            os.makedirs(label_dir) 
        
        # Save the image as a PNG file
        img_filename = os.path.join(label_dir, f"{i}.png")
        plt.imsave(img_filename, img, cmap='gray')

save_dir = '/WAVE/users2/unix/klor/ycho_lab/kimsong_lor/csen_319/CuKNN/data/training_data'
save_images_as_png(x_train, y_train, save_dir)
save_dir = '/WAVE/users2/unix/klor/ycho_lab/kimsong_lor/csen_319/CuKNN/data/testing_data'
save_images_as_png(x_test, y_test, save_dir)
