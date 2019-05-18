from keras.models import load_model
import keras.backend as K
import sys
from keras.models import Model
import cv2
import numpy as np

import matplotlib.pyplot as plt

def precision(y_true, y_pred): #taken from old keras source code
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
     precision = true_positives / (predicted_positives + K.epsilon())
     return precision
def recall(y_true, y_pred): #taken from old keras source code
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
     recall = true_positives / (possible_positives + K.epsilon())
     return recall
def get_model():
   return load_model('model_vgg16_2.h5',custom_objects={'precision':precision,'recall':recall})