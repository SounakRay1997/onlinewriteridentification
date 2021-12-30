import os
import random
import keras
import pickle 
import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from keras.models import load_model, Model
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.layers import Layer

class lstm_bottleneck(Layer):
    def __init__(self, lstm_units, time_steps, **kwargs):
        self.lstm_units = lstm_units
        self.time_steps = time_steps
        self.lstm_layer = LSTM(lstm_units, return_sequences=False)
        self.repeat_layer = RepeatVector(time_steps)
        super(lstm_bottleneck, self).__init__(**kwargs)
    def call(self, inputs):
        return self.repeat_layer(self.lstm_layer(inputs))
    def compute_mask(self, inputs, mask=None):
        return mask
