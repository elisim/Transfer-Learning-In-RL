import keras
import tensorflow as tf
from keras.layers import Concatenate
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

from actor_softmax_keras import ActorNetworkSoftmax
from utils import StateScaler


def actor_loss(output, target, advantage):
    return keras.losses.mean_squared_error(target, output) * advantage


class ActorNetworkRegressor:
    def __init__(self, state_size, action_size, learning_rate, game_action_size, env, is_scale=False):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.action_size = action_size
        self.game_action_size = game_action_size
        self.is_scale = is_scale
        self.scaller = StateScaler(env)
        self.transfer_model = None
        self.transfer_model_pred = None

        # This returns a tensor
        inputs = Input(shape=(self.state_size,))
        y_true = Input(shape=(1,), name='y_true')
        advantage = Input(shape=(self.action_size,), name='is_weight')

        # a layer instance is callable on a tensor, and returns a tensor
        output_1 = Dense(12, activation='relu', name="d1",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         bias_initializer=tf.zeros_initializer())(inputs)
        output_2 = Dense(12, activation='relu', name="d2",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         bias_initializer=tf.zeros_initializer())(output_1)
        predictions = Dense(1, activation='linear')(output_2)

        # This creates a model that includes
        # the Input layer and three Dense layers
        self.policy = Model(inputs=[inputs, y_true, advantage], outputs=predictions)
        self.policy.add_loss(actor_loss(y_true, predictions, advantage=advantage))
        self.policy.compile(loss=None, optimizer=Adam(lr=self.learning_rate))

        self.policy_pred = Model(inputs=inputs, outputs=predictions, name='test_only')

    def create_progressive_network(self, models_to_load, input_dict):
        models = []

        for model in models_to_load:
            model_to_freeze = ActorNetworkSoftmax(input_dict['network_state_size'], input_dict['network_action_size'],
                                                  input_dict['learning_rate_p'], input_dict['game_action_size'],
                                                  input_dict['env'])
            model_to_freeze.loading(model)
            for layer in model_to_freeze.actor.layers:
                layer.trainable = False
            models.append(model_to_freeze.actor)

        concat_layer = Concatenate()(
            [self.policy.layers[1].output, models[0].layers[1].output, models[1].layers[1].output])
        output_layer = Dense(1, activation='linear')(concat_layer)

        # This creates a model that includes
        # the Input layer and three Dense layers
        y_true = Input(shape=(1,), name='y_true')
        advantage = Input(shape=(self.action_size,), name='is_weight')

        model = Model(inputs=[self.policy.input[0], models[0].input[0], models[1].input[0], y_true, advantage],
                      outputs=output_layer)

        model.add_loss(actor_loss(y_true, output_layer, advantage=advantage))
        model.compile(loss=None, optimizer=Adam(lr=self.learning_rate))

        model_pred = Model(inputs=[self.policy.input[0], models[0].input[0], models[1].input[0]], outputs=output_layer,
                           name='test_only')

        self.transfer_model = model
        self.transfer_model_pred = model_pred
        pass

    def __call__(self):
        return self.policy

    def fitting(self, state, action, adventage):

        if self.is_scale:
            state = self.scaller.scale(state)

        if self.transfer_model is not None:
            self.transfer_model.fit([state, state, state, action, adventage], epochs=1, verbose=0)
        else:
            self.policy.fit([state, action, adventage], epochs=1, verbose=0)

    def predicting(self, state):
        if self.is_scale:
            state = self.scaller.scale(state)

        if self.transfer_model is not None:
            action = self.transfer_model_pred.predict([state, state, state])
        else:
            action = self.policy_pred.predict(state)

        return action

    def saving(self, path):
        self.policy.save_weights(path)

    def loading(self, path, input_dict):
        model_to_load = ActorNetworkSoftmax(input_dict['state_size'], input_dict['action_size'],
                                            input_dict['learning_rate_p'], input_dict['game_action_size'],
                                            input_dict['env'])
        model_to_load.loading(path)

        weights_list = model_to_load.actor.get_weights()
        for i in range(2, 4, 2):
            self.policy.layers[i].set_weights([weights_list[i], weights_list[i + 1]])
