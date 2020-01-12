import time
from collections import deque
from math import floor
import gym

import numpy as np
import tensorflow as tf


class ActorCriticTrainer:
    def __init__(self, model_config):
        input_dict = model_config.params
        self.max_episodes = input_dict['max_episodes']
        self.max_steps = input_dict['max_steps']
        self.network_state_size = input_dict['network_state_size']
        self.network_action_size = input_dict['network_action_size']
        self.env = gym.make(model_config.name)
        self.discount_factor = input_dict['discount_factor']
        self.game_action_size = input_dict['game_action_size']

        #
        self.policy = model_config.actor
        self.critic = model_config.critic

    def load_prog_network(self, models_list, input_dict):
        models_to_load = [model_to_load + "_critic.model" for model_to_load in models_list]
        self.critic.load_prog_network(models_to_load, input_dict)

        models_to_load = [model_to_load + "_actor.model" for model_to_load in models_list]
        self.policy.load_prog_network(models_to_load, input_dict)

        pass

    def save(self, path):
        critic_path = path + "_critic.model"
        self.critic.saving(critic_path)

        actor_path = path + "_actor.model"
        self.policy.saving(actor_path)

    def load(self, path, input_dict):

        critic_path = path + "_critic.model"
        self.critic.loading(critic_path)

        actor_path = path + "_actor.model"
        self.policy.loading(actor_path, input_dict)

    def train(self):
        # Start training the agent with REINFORCE algorithm
        solved = False
        episode_rewards = np.zeros(self.max_episodes)
        average_rewards = 0.0
        avg_episode_rewards = deque(maxlen=100)
        results_dict = {'episode': [], 'reward': [], 'average_rewards': [], 'time': []}

        for episode in range(self.max_episodes):
            state = self.env.reset()
            state = np.pad(state, (0, self.network_state_size - len(state)))
            state = state.reshape([1, self.network_state_size])
            start_time = time.time()

            for step in range(self.max_steps):

                target = np.zeros((1, 1))
                advantages = np.zeros((1, self.network_action_size))

                action = self.policy.predicting(state)  # Get pi(s)

                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape([len(next_state)])
                next_state = np.pad(next_state, (0, self.network_state_size - len(next_state)))

                next_state = next_state.reshape([1, self.network_state_size])

                episode_rewards[episode] += reward

                # Predict with baseline and update advantage
                value_curr = self.critic.predicting(state)
                value_next = self.critic.predicting(next_state)

                if done:  # True if S' is terminal
                    advantages[0][floor(abs(action))] = reward - value_curr
                    target[0][0] = reward

                else:
                    advantages[0][floor(abs(action))] = reward + self.discount_factor * value_next - value_curr
                    target[0][0] = reward + self.discount_factor * value_next

                # Update critic
                self.critic.fitting(state, target)

                # Update policy
                self.policy.fitting(state, action, advantages)

                if done:
                    if round(np.mean(avg_episode_rewards), 2) > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break

                state = next_state

            avg_episode_rewards.append(episode_rewards[episode])
            results_dict['episode'].append(episode)
            results_dict['reward'].append(episode_rewards[episode])
            results_dict['average_rewards'].append(round(np.mean(avg_episode_rewards), 2))
            results_dict['time'].append(round(time.time() - start_time, 2))
            print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                               round(np.mean(
                                                                                   avg_episode_rewards),
                                                                                   2)))

            if solved:
                break
