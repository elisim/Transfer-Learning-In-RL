import numpy as np
import keras
from keras.layers import Input, Dense, Concatenate
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.models import Model
from keras.optimizers import Adam

from utils import StateScaler


def regression_loss(output, target, advantage):
    return mean_squared_error(target, output) * advantage


def classification_loss(output, target, advantage):
    return categorical_crossentropy(target, output) * advantage


class ActorNetworkSoftmax:
    def __init__(self, network_state_size, network_action_size, learning_rate, game_action_size, env,
                 should_scale=False):
        self.network_state_size = network_state_size
        self.learning_rate = learning_rate
        self.network_action_size = network_action_size
        self.game_action_size = game_action_size
        self.should_scale = should_scale
        self.scaler = StateScaler(env)

        self.transfer_model = None
        self.transfer_model_pred = None

        # TODO: Switch to this:
        # self.actor = keras.Sequential()
        # self.actor.add(Dense(12, input_dim=self.network_state_size, activation='relu', kernel_initializer='he_uniform'))
        # self.actor.add(Dense(12, activation='relu', kernel_initializer='he_uniform'))
        # self.actor.add(Dense(self.network_action_size, activation='softmax', kernel_initializer='he_uniform'))
        # self.actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        # print("Actor model:")
        # self.actor.summary()

        # This returns a tensor
        inputs = Input(shape=(self.network_state_size,))
        y_true = Input(shape=(self.network_action_size,), name='y_true')
        advantage = Input(shape=(self.network_action_size,), name='is_weight')

        # a layer instance is callable on a tensor, and returns a tensor
        output_1 = Dense(16, activation='relu')(inputs)
        output_2 = Dense(16, activation='relu')(output_1)
        predictions = Dense(self.network_action_size, activation='softmax')(output_2)

        # This creates a model that includes
        # the Input layer and three Dense layers
        self.actor = Model(inputs=[inputs, y_true, advantage], outputs=predictions)
        self.actor.add_loss(classification_loss(y_true, predictions, advantage=advantage))
        self.actor.compile(loss=None, optimizer=Adam(lr=self.learning_rate))

        self.actor_pred = Model(inputs=inputs, outputs=predictions, name='test_only')

    def create_progressive_network(self, models_to_load, params):
        """
        Create a progressive network out of the given models
        :param models_to_load: models to be used in the network
        """
        models = []

        for model in models_to_load:
            model_to_freeze = ActorNetworkSoftmax(params['network_state_size'], params['network_action_size'],
                                                  params['learning_rate_p'], params['game_action_size'], params['env'])
            model_to_freeze.loading(model)
            for layer in model_to_freeze.actor.layers:
                layer.trainable = False
            models.append(model_to_freeze.actor)

        # Contact layers of the models to create the progressive network
        concat_layer = Concatenate()(
            [self.actor.layers[1].output, models[0].layers[1].output, models[1].layers[1].output])
        output_layer = Dense(self.network_action_size, activation='softmax')(concat_layer)

        # This creates a model that includes
        # the Input layer and three Dense layers
        y_true = Input(shape=(self.network_action_size,), name='y_true')
        advantage = Input(shape=(self.network_action_size,), name='is_weight')

        model = Model(inputs=[self.actor.input[0], models[0].input[0], models[1].input[0], y_true, advantage],
                      outputs=output_layer)

        model.add_loss(classification_loss(y_true, output_layer, advantage=advantage))
        model.compile(loss=None, optimizer=Adam(lr=self.learning_rate))

        model_pred = Model(inputs=[self.actor.input[0], models[0].input[0], models[1].input[0]], outputs=output_layer,
                           name='test_only')

        self.transfer_model = model
        self.transfer_model_pred = model_pred

    def __call__(self):
        return self.actor

    def fitting(self, state, action, adventage):
        action_one_hot = np.zeros((1, self.network_action_size))
        action_one_hot[0, action] = 1

        if self.should_scale:
            state = self.scaler.scale(state)
        if self.transfer_model is not None:
            self.transfer_model.fit([state, state, state, action_one_hot, adventage], epochs=1, verbose=0)
        else:
            self.actor.fit([state, action_one_hot, adventage], epochs=1, verbose=0)

    def predicting(self, state):
        if self.should_scale:
            state = self.scaler.scale(state)

        if self.transfer_model is not None:
            actions_distribution = self.transfer_model_pred.predict([state, state, state])[0]
        else:
            actions_distribution = self.actor_pred.predict(state)[0]

        action = np.inf
        while action > self.game_action_size - 1:
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)

        return action

    def saving(self, path):
        self.actor.save_weights(path)

    def loading(self, path, input_dict=None):
        self.actor.load_weights(path)


class ActorNetworkRegressor:
    def __init__(self, state_size, action_size, learning_rate, game_action_size, env, is_scale=False):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.action_size = action_size
        self.game_action_size = game_action_size
        self.is_scale = is_scale
        self.scaler = StateScaler(env)
        self.transfer_model = None
        self.transfer_model_pred = None

        # This returns a tensor
        inputs = Input(shape=(self.state_size,))
        y_true = Input(shape=(1,), name='y_true')
        advantage = Input(shape=(self.action_size,), name='is_weight')

        # a layer instance is callable on a tensor, and returns a tensor
        output_1 = Dense(12, activation='relu', name="d1",
                         kernel_initializer=keras.initializers.glorot_uniform(),
                         bias_initializer=keras.initializers.Zeros())(inputs)
        output_2 = Dense(12, activation='relu', name="d2",
                         kernel_initializer=keras.initializers.glorot_uniform(),
                         bias_initializer=keras.initializers.Zeros())(output_1)
        predictions = Dense(1, activation='linear')(output_2)

        # This creates a model that includes
        # the Input layer and three Dense layers
        self.policy = Model(inputs=[inputs, y_true, advantage], outputs=predictions)
        self.policy.add_loss(regression_loss(y_true, predictions, advantage=advantage))
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

        model.add_loss(regression_loss(y_true, output_layer, advantage=advantage))
        model.compile(loss=None, optimizer=Adam(lr=self.learning_rate))

        model_pred = Model(inputs=[self.policy.input[0], models[0].input[0], models[1].input[0]], outputs=output_layer,
                           name='test_only')

        self.transfer_model = model
        self.transfer_model_pred = model_pred

    def __call__(self):
        return self.policy

    def fitting(self, state, action, advantage):
        if self.is_scale:
            state = self.scaler.scale(state)

        if self.transfer_model is not None:
            self.transfer_model.fit([state, state, state, action, advantage], epochs=1, verbose=0)
        else:
            self.policy.fit([state, action, advantage], epochs=1, verbose=0)

    def predicting(self, state):
        if self.is_scale:
            state = self.scaler.scale(state)

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
