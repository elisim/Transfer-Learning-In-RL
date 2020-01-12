import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, Concatenate
from keras.models import Sequential
from keras.optimizers import Adam


class CriticNetwork:
    def __init__(self, state_size, learning_rate):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.critic = Sequential()
        self.critic.add(Dense(units=32, input_dim=self.state_size, activation='relu'))
        self.critic.add(Dense(units=32, activation='relu'))
        self.critic.add(Dense(units=32, activation='relu'))
        self.critic.add(Dense(units=1))
        self.critic.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.transfer_model = None

    def __call__(self):
        return self.critic

    def fitting(self, state, target):

        if self.transfer_model is not None:
            self.transfer_model.fit([state, state, state], target, epochs=1, verbose=0)
        else:
            self.critic.fit(state, target, epochs=1, verbose=0)

    def predicting(self, state):
        if self.transfer_model is not None:
            res = self.transfer_model.predict([state, state, state])
        else:
            res = self.critic.predict(state)[0]
        return res

    def load_prog_network(self, models_to_load, input_dict):
        freezed_layers = []
        models = []

        for model in models_to_load:
            model_to_freeze = CriticNetwork(input_dict['state_size'], input_dict['learning_rate_v'], input_dict['env'])
            model_to_freeze.loading(model)
            for layer in model_to_freeze.critic.layers:
                layer.trainable = False
            models.append(model_to_freeze.critic)

        concat_layer = Concatenate()(
            [self.critic.layers[1].output, models[0].layers[1].output, models[1].layers[1].output])
        output_layer = Dense(units=1)(concat_layer)
        model = Model([self.critic.input, models[0].input, models[1].input], output_layer)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.transfer_model = model
        pass

    def saving(self, path):
        self.critic.save_weights(path)

    def loading(self, path):
        self.critic.load_weights(path)
