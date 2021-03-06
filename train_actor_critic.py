from collections import deque
from math import floor
from utils import TensorFlowLogger
import numpy as np


class ActorCriticTrainer:
    def __init__(self, input_dict, unfreeze_layers_at=None):
        self.max_episodes = input_dict['max_episodes']
        self.max_steps = input_dict['max_steps']
        self.state_size = input_dict['network_state_size']
        self.action_size = input_dict['network_action_size']
        self.env = input_dict['env']
        self.render = input_dict['render']
        self.discount_factor = input_dict['discount_factor']
        self.game_action_size = input_dict['game_action_size']
        self.solve_threshold = input_dict['solve_threshold']
        self.actor = input_dict['actor']
        self.critic = input_dict['critic']
        self.tf_logger = TensorFlowLogger()
        self.unfreeze_layers_at = unfreeze_layers_at

    def create_progressive_network(self, models_list, params):
        models_to_load = [f'models/{model_to_load}_critic.model' for model_to_load in models_list]
        self.critic.create_progressive_network(models_to_load, params)

        models_to_load = [f'models/{model_to_load}_actor.model' for model_to_load in models_list]
        self.actor.create_progressive_network(models_to_load, params)

    def save(self, path):
        critic_path = path + "_critic.model"
        self.critic.saving_weights(critic_path)

        actor_path = path + "_actor.model"
        self.actor.saving_weights(actor_path)

    def load(self, path, input_dict):
        critic_path = path + "_critic.model"
        self.critic.loading_weights(critic_path)

        actor_path = path + "_actor.model"
        self.actor.loading_weights(actor_path, input_dict)

    def train(self):
        solved = False
        episode_rewards = np.zeros(self.max_episodes)
        last_100_rewards = deque(maxlen=100)
        total_steps = 0
        if self.unfreeze_layers_at is not None:
            self.actor.freeze_layers()
            self.critic.freeze_layers()

        for episode in range(self.max_episodes):
            if self.unfreeze_layers_at == episode:
                self.actor.unfreeze_layers()
                self.critic.unfreeze_layers()
            state = self.env.reset()
            state = np.pad(state, (0, self.state_size - len(state)))
            state = state.reshape([1, self.state_size])

            for step in range(self.max_steps):
                total_steps += 1
                target = np.zeros((1, 1))
                advantages = np.zeros((1, self.action_size))

                action = self.actor.predicting(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape([len(next_state)])
                next_state = np.pad(next_state, (0, self.state_size - len(next_state)))

                next_state = next_state.reshape([1, self.state_size])

                if self.render:
                    self.env.render()

                episode_rewards[episode] += reward

                value_curr = self.critic.predicting(state)
                value_next = self.critic.predicting(next_state)

                if done:
                    advantages[0][floor(abs(action))] = reward - value_curr
                    target[0][0] = reward

                else:
                    advantages[0][floor(abs(action))] = reward + self.discount_factor * value_next - value_curr
                    target[0][0] = reward + self.discount_factor * value_next

                # Update critic
                self.critic.fitting(state, target)

                # Update actor
                self.actor.fitting(state, action, advantages)

                if done:
                    if round(np.mean(last_100_rewards), 2) > self.solve_threshold:
                        solved = True
                    break

                state = next_state

            last_100_rewards.append(episode_rewards[episode])
            avg_rewards = round(np.mean(last_100_rewards), 2)
            self.tf_logger.log_scalar(tag='average_100_episodes_reward', value=avg_rewards, step=episode)
            print(
                f'Episode {episode}, Number of Steps: {step}, Reward: {episode_rewards[episode]:.2f} Average over 100 episodes: {avg_rewards}')

            if solved:
                print('Solved at episode: ' + str(episode))
                print('Total Steps: ' + str(total_steps))
                break
