import time
from collections import deque

import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import keras
from keras.layers import Input, Dense
from keras.models import Model

from actor_regressor import StateScaler


def my_loss(output, target, adventage):
    return keras.losses.categorical_crossentropy(target, output) * adventage


class ActorNetworkSoftmaxKeras:
    def __init__(self, state_size, action_size, learning_rate, game_action_size, env, is_scale=False):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.action_size = action_size
        self.game_action_size = game_action_size
        self.is_scale = is_scale
        self.scaller = StateScaler(env)

        # This returns a tensor
        inputs = Input(shape=(self.state_size,))
        y_true = Input(shape=(self.action_size,), name='y_true')
        advantage = Input(shape=(self.action_size,), name='is_weight')

        # a layer instance is callable on a tensor, and returns a tensor
        output_1 = Dense(12, activation='relu', name="d1",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         bias_initializer=tf.zeros_initializer())(inputs)
        output_2 = Dense(12, activation='relu', name="d2",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         bias_initializer=tf.zeros_initializer())(output_1)
        predictions = Dense(self.action_size, activation='softmax')(output_2)

        # This creates a model that includes
        # the Input layer and three Dense layers
        self.policy = Model(inputs=[inputs, y_true, advantage], outputs=predictions)
        self.policy.add_loss(my_loss(y_true, predictions, adventage=advantage))
        self.policy.compile(loss=None, optimizer=Adam(lr=self.learning_rate))

        self.policy_pred = Model(inputs=inputs, outputs=predictions, name='test_only')

    def __call__(self):
        return self.policy

    def fitting(self, state, action, adventage):
        action_one_hot = np.zeros((1, self.action_size))
        action_one_hot[0, action] = 1
        # self.policy.compile(loss=custom_loss(adventage), optimizer=Adam(lr=self.learning_rate))

        if self.is_scale:
            state = self.scaller.scale(state)
        # state = state.reshape([1, self.state_size])

        self.policy.fit([state, action_one_hot, adventage], epochs=1, verbose=0)

    def predicting(self, state):
        if self.is_scale:
            state = self.scaller.scale(state)
        actions_distribution = self.policy_pred.predict(state)[0]
        action = np.inf
        while action > self.game_action_size - 1:
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)

        return action

    def saving(self, path):
        self.policy.save_weights(path)

    def loading(self, path):
        self.policy.load_weights(path)
