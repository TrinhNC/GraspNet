# -*- coding: utf-8 -*-

""" Based on AlexNet.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
"""

from __future__ import print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.metrics import *
import pickle
import torchfile
import numpy as np
import tensorflow as tf
from grasp_metric import *

def rect_metric(prediction, target, inputs):   
	return tf.reduce_mean(grasp_error(prediction,target))
'''
def rect_metric(prediction, target, inputs):
	return tf.reduce_sum(grasp_error(prediction,target), name='rectme') 
'''
X_train, Y_train = torchfile.load('train.t7')
X_val, Y_val = torchfile.load('val.t7')
X_train = X_train.transpose(0, 2, 3, 1)
X_val = X_val.transpose(0, 2, 3, 1)

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 2048, activation='relu') #activation='tanh'         
network = dropout(network, 0.5)
network = fully_connected(network, 2048, activation='relu') #activation='tanh'
network = dropout(network, 0.5)
network = fully_connected(network, 6, activation='linear', name='fc')
#rect_metric = tflearn.metrics.Rectangle()
network = regression(network, optimizer='momentum',metric=rect_metric,
                     loss='mean_square',
                     learning_rate=0.0005)	
        
# Training
model = tflearn.DNN(network, checkpoint_path='grasp_net', best_checkpoint_path='grasp_net_best',
                    max_checkpoints=1, tensorboard_verbose=0)
						
#model.load("model_alexnet-10062")
                    
model.fit(X_train, Y_train, n_epoch=1000, validation_set=0.1, 
          shuffle=True, show_metric=True, batch_size=128, 
          snapshot_epoch=True, run_id='grasp_net')
          
model.save("grasp_net.tfl")
print("Network trained and saved as grasp_net.tfl!")


