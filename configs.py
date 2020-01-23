import gym

from actor import ActorNetworkSoftmax, ActorNetworkRegressor
from critic import CriticNetwork


def get_params(game):
    game_action_size = 0

    if game == 'CartPole-v1':
        game_action_size = 2
    elif game == 'Acrobot-v1':
        game_action_size = 3
    elif game == 'MountainCarContinuous-v0':
        game_action_size = 1

    input_dict = {
        'env': gym.make(game),
        'network_state_size': 6,
        'network_action_size': 3,
        'game_action_size': game_action_size,
        'max_episodes': 5000,
        'max_steps': 500,
        'discount_factor': 0.99,
        'learning_rate_actor': 0.0004,
        'learning_rate_value': 0.0004,
        'render': False
    }

    # Initialize the actor network

    if game == 'CartPole-v1':
        input_dict['solve_threshold'] = 475
        input_dict['actor'] = ActorNetworkSoftmax(input_dict['network_state_size'], input_dict['network_action_size'],
                                                  input_dict['learning_rate_actor'], input_dict['game_action_size'],
                                                  input_dict['env'])
        input_dict['critic'] = CriticNetwork(input_dict['network_state_size'], input_dict['learning_rate_value'],
                                             input_dict['env'])

    elif game == 'Acrobot-v1':
        input_dict['solve_threshold'] = -90
        input_dict['actor'] = ActorNetworkSoftmax(input_dict['network_state_size'], input_dict['network_action_size'],
                                                  input_dict['learning_rate_actor'], input_dict['game_action_size'],
                                                  input_dict['env'])
        input_dict['critic'] = CriticNetwork(input_dict['network_state_size'], input_dict['learning_rate_value'],
                                             input_dict['env'])

    elif game == 'MountainCarContinuous-v0':
        input_dict['solve_threshold'] = 90
        input_dict['max_steps'] = 5000
        input_dict['learning_rate_actor'] = 0.00004
        input_dict['actor'] = ActorNetworkRegressor(input_dict['network_state_size'],
                                                    input_dict['network_action_size'],
                                                    input_dict['learning_rate_actor'], input_dict['game_action_size'],
                                                    input_dict['env'], is_scale=True)
        input_dict['critic'] = CriticNetwork(input_dict['network_state_size'], input_dict['learning_rate_value'],
                                             input_dict['env'],
                                             is_scale=True)

    return input_dict
