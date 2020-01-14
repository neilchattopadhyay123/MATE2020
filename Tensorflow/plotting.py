import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
from tensorflow import keras

fashion_mnist= keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.imshow(test_images[1], cmap= plt.cm.get_cmap("binary"))
plt.show()
