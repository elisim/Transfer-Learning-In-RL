import gym

from actor_regressor_keras import ActorNetworkRegressor
from actor_softmax_keras import ActorNetworkSoftmax
from critic import CriticNetwork


def get_params(game):
    game_action_size = 0

    if game == 'CartPole-v1':
        env = gym.make('CartPole-v1')
        game_action_size = 2
    elif game == 'Acrobot-v1':
        env = gym.make('Acrobot-v1')
        game_action_size = 3
    elif game == 'MountainCarContinuous-v0':
        env = gym.make('MountainCarContinuous-v0')
        game_action_size = 1

    input_dict = {
        'env': env,
        'network_state_size': 6,
        'network_action_size': 3,
        'game_action_size': game_action_size,
        'max_episodes': 1000,
        'max_steps': 500,
        'discount_factor': 0.99,
        'learning_rate_p': 0.0004,
        'learning_rate_v': 0.0004,
        'render': False,
    }

    # Initialize the policy network

    if game == 'CartPole-v1':
        input_dict['solve_threshold'] = 475
        input_dict['policy'] = ActorNetworkSoftmax(input_dict['network_state_size'], input_dict['network_action_size'],
                                                   input_dict['learning_rate_p'], input_dict['game_action_size'],
                                                   input_dict['env'])
        input_dict['critic'] = CriticNetwork(input_dict['network_state_size'], input_dict['learning_rate_v'],
                                             input_dict['env'])

    elif game == 'Acrobot-v1':
        input_dict['solve_threshold'] = -90
        input_dict['policy'] = ActorNetworkSoftmax(input_dict['network_state_size'], input_dict['network_action_size'],
                                                   input_dict['learning_rate_p'], input_dict['game_action_size'],
                                                   input_dict['env'])
        input_dict['critic'] = CriticNetwork(input_dict['network_state_size'], input_dict['learning_rate_v'],
                                             input_dict['env'])

    elif game == 'MountainCarContinuous-v0':
        input_dict['policy'] = ActorNetworkRegressor(input_dict['network_state_size'],
                                                     input_dict['network_action_size'],
                                                     input_dict['learning_rate_p'], input_dict['game_action_size'],
                                                     input_dict['env'], is_scale=True)
        input_dict['critic'] = CriticNetwork(input_dict['network_state_size'], input_dict['learning_rate_v'],
                                             input_dict['env'],
                                             is_scale=True)

    return input_dict
