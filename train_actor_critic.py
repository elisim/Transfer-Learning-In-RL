import time
from collections import deque
from math import floor

import numpy as np
import pandas as pd


class ActorCriticTrainer:
    def __init__(self, input_dict):
        self.max_episodes = input_dict['max_episodes']
        self.max_steps = input_dict['max_steps']
        self.state_size = input_dict['network_state_size']
        self.action_size = input_dict['network_action_size']
        self.env = input_dict['env']
        self.render = input_dict['render']
        self.discount_factor = input_dict['discount_factor']
        self.game_action_size = input_dict['game_action_size']

        if 'solve_threshold' in input_dict:
            self.solve_threshold = input_dict['solve_threshold']
        else:
            self.solve_threshold = 1000

        #
        self.policy = input_dict['policy']
        self.critic = input_dict['critic']

    def create_progressive_network(self, models_list, params):
        models_to_load = [model_to_load + "_critic.model" for model_to_load in models_list]
        self.critic.create_progressive_network(models_to_load, params)

        models_to_load = [model_to_load + "_actor.model" for model_to_load in models_list]
        self.policy.create_progressive_network(models_to_load, params)

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
        last_100_rewards = deque(maxlen=100)
        results_dict = {'episode': [], 'reward': [], 'average_rewards': [], 'time': []}

        for episode in range(self.max_episodes):
            state = self.env.reset()
            state = np.pad(state, (0, self.state_size - len(state)))
            state = state.reshape([1, self.state_size])
            start_time = time.time()

            for step in range(self.max_steps):

                target = np.zeros((1, 1))
                advantages = np.zeros((1, self.action_size))

                action = self.policy.predicting(state)  # Get pi(s)

                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape([len(next_state)])
                next_state = np.pad(next_state, (0, self.state_size - len(next_state)))

                next_state = next_state.reshape([1, self.state_size])

                if self.render:
                    self.env.render()

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
                    if round(np.mean(last_100_rewards), 2) > self.solve_threshold:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break

                state = next_state

            avg_rewards = round(np.mean(last_100_rewards), 2)
            last_100_rewards.append(episode_rewards[episode])
            results_dict['episode'].append(episode)
            results_dict['reward'].append(episode_rewards[episode])
            results_dict['average_rewards'].append(avg_rewards)
            results_dict['time'].append(round(time.time() - start_time, 2))
            print(f'Episode {episode}, Number of Steps: {step}, Reward: {episode_rewards[episode]} Average over 100 episodes: {avg_rewards}')

            if solved:
                break

        pd.DataFrame(results_dict).to_csv('actor_critic_results2.csv')
