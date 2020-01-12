import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam


def my_loss(output, target, adventage):
    return keras.losses.categorical_crossentropy(target, output) * adventage


class ActorNetworkSoftmax:
    def __init__(self, state_size, network_action_size, learning_rate, game_action_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.network_action_size = network_action_size
        self.game_action_size = game_action_size
        self.transfer_model = None
        self.transfer_model_pred = None

        self.policy = keras.Sequential()
        self.policy.add(Dense(12, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        self.policy.add(Dense(12, activation='relu', kernel_initializer='he_uniform'))
        self.policy.add(Dense(self.network_action_size, activation='softmax', kernel_initializer='he_uniform'))
        self.policy.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        print("Actor model:")
        self.policy.summary()


    # def load_prog_network(self, models_to_load, input_dict):
    #     freezed_layers = []
    #     models = []
    #
    #     for model in models_to_load:
    #         model_to_freeze = ActorNetworkSoftmaxKeras(input_dict['state_size'], input_dict['action_size'], input_dict['learning_rate_p'], input_dict['game_action_size'], input_dict['env'])
    #         model_to_freeze.loading(model)
    #         for layer in model_to_freeze.policy.layers:
    #             layer.trainable = False
    #         models.append(model_to_freeze.policy)
    #
    #     concat_layer = Concatenate()(
    #         [self.policy.layers[1].output, models[0].layers[1].output, models[1].layers[1].output])
    #     output_layer = Dense(self.action_size, activation='softmax')(concat_layer)
    #
    #     # This creates a model that includes
    #     # the Input layer and three Dense layers
    #     y_true = Input(shape=(self.action_size,), name='y_true')
    #     advantage = Input(shape=(self.action_size,), name='is_weight')
    #
    #     model = Model(inputs=[self.policy.input[0], models[0].input[0], models[1].input[0], y_true, advantage],
    #                   outputs=output_layer)
    #
    #     model.add_loss(my_loss(y_true, output_layer, adventage=advantage))
    #     model.compile(loss=None, optimizer=Adam(lr=self.learning_rate))
    #
    #     model_pred = Model(inputs=[self.policy.input[0], models[0].input[0], models[1].input[0]], outputs=output_layer, name='test_only')
    #
    #     self.transfer_model = model
    #     self.transfer_model_pred = model_pred
    #     pass
    #
    # def __call__(self):
    #     return self.policy

    def fitting(self, state, action, adventage):
        self.policy.fit(state, adventage, epochs=1, verbose=0)

    def predicting(self, state):
        if self.transfer_model is not None:
            actions_distribution = self.transfer_model_pred.predict([state, state, state])[0]
        else:
            actions_distribution = self.policy.predict(state)[0]

        action = np.inf
        while action >= self.game_action_size:
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)

        return action

    def saving(self, path):
        self.policy.save_weights(path)

    def loading(self, path):
        self.policy.load_weights(path)
