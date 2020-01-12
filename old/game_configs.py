from softmax_actor import ActorNetworkSoftmax
import gym
from critic import CriticNetwork


class GameConfig:
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.actor = ActorNetworkSoftmax(params['network_state_size'],
                                         params['network_action_size'],
                                         params['learning_rate_p'],
                                         params['game_action_size'])
        self.critic = CriticNetwork(params['network_state_size'], params['learning_rate_v'])


# car = 'MountainCarContinuous-v0'

cartpole = GameConfig('CartPole-v1', {
    'network_state_size': 6,
    'network_action_size': 3,
    'game_action_size': 2,
    'max_episodes': 1000,
    'max_steps': 500,
    'discount_factor': 0.99,
    'learning_rate_p': 0.0004,
    'learning_rate_v': 0.0004,
})

acrobot = GameConfig('Acrobot-v1', {
    'network_state_size': 6,
    'network_action_size': 3,
    'game_action_size': 3,
    'max_episodes': 1000,
    'max_steps': 50000,
    'discount_factor': 0.99,
    'learning_rate_p': 0.000004,
    'learning_rate_v': 0.0004,
})
