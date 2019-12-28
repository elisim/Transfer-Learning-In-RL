# https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
import gym
import numpy as np
import tensorflow as tf
import datetime

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import keras.backend as K


np.random.seed(1)
env = gym.make('MountainCarContinuous-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]


class TensorFlowLogger:
    def __init__(self, with_baseline):
        self._log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         if with_baseline:
#             self._log_dir += "_with_baseline"
        self._file_writer = tf.summary.FileWriter(self._log_dir + "/metrics")

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._file_writer.add_summary(summary, step)


class CriticNetwork:
    """
    Value Network
    """
    def __init__(self, learning_rate):
        self.critic = Sequential()
        self.critic.add(Dense(units=32, input_dim=state_size, activation='elu'))
        self.critic.add(Dense(units=32, activation='elu'))
        self.critic.add(Dense(units=1))
        self.critic.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    def __call__(self):
        return self.critic


def PolicyNetwork():
    """
    The network input is the state and output are two scalar functions, μ(s) and σ(s),
    which are used as the mean and standard deviation of a Gaussian (normal) distribution.
    We will choose our actions by sampling from this distribution.
    """
    def get_action(state):
        with tf.variable_scope("policy_network"):
            init_xavier = tf.contrib.layers.xavier_initializer()

            hidden1 = tf.layers.dense(state, 32, tf.nn.elu, init_xavier)
            hidden2 = tf.layers.dense(hidden1, 32, tf.nn.elu, init_xavier)
            mu = tf.layers.dense(hidden2, 1, init_xavier)
            sigma = tf.layers.dense(hidden2, 1, tf.nn.softplus, init_xavier)
            sigma = tf.nn.softplus(sigma) + 1e-5

            norm_dist = tf.contrib.distributions.Normal(mu, sigma)
            action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
            action_tf_var = tf.clip_by_value(action_tf_var, env.action_space.low[0], env.action_space.high[0])

        return action_tf_var, norm_dist

    def minimize():
        # define actor (policy) loss function
        loss_actor = -tf.log(norm_dist.prob(action_placeholder) + 1e-5) * delta_placeholder
        training_op_actor = tf.train.AdamOptimizer(
            lr_actor, name='actor_optimizer').minimize(loss_actor)


def main():
    # Define hyperparameters
    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    actor_learning_rate = 0.0004
    critic_learning_rate = 0.0004
    render = False

    # Create Logger to log scalars
    tf_logger = TensorFlowLogger()

    # actor critic
    action_tf_var, norm_dist = policy_network(state_placeholder)
    critic = CriticNetwork(critic_learning_rate)()

    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    total_steps = 0

    for episode in range(max_episodes):
        steps_per_episode = 0
        state = env.reset()
        episode_transitions = []

        # Generate an episode, out is list of episode_transitions: state, action, reward, next_state and done.
        for step in range(max_steps):

            # action.shape = (1,1)
            # action = actor.get_action(state) #### TODO

            # Execute action and observe reward & next state from env
            next_state, reward, done, _ = env.step(action)
            episode_rewards[episode] += reward

            # V_of_next_state.shape=(1,1)
            V_of_next_state = critic.predict(next_state)

            # Set TD Target: target = r + gamma * V(next_state)
            target = reward + discount_factor * np.squeeze(V_of_next_state)

            # td_error = target - V(s)
            V_of_curr_state = critic.predict(state)
            td_error = target - np.squeeze(V_of_curr_state)

            # Update actor
            actor.model.fit()

            # Update critic
            critic.fit(target, V_of_curr_state)


            # Update actor by minimizing loss (Actor training)
            _, loss_actor_val = sess.run(
                [training_op_actor, loss_actor],
                feed_dict={action_placeholder: np.squeeze(action),
                           state_placeholder: scale_state(state),
                           delta_placeholder: td_error})
            if render:
                env.render()

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                print(
                    "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    print(' Total Steps: ' + str(total_steps))
                    solved = True
                break
            state = next_state
            steps_per_episode += 1

        tf_logger.log_scalar(tag='average_100_episodes_reward', value=average_rewards, step=episode)
        tf_logger.log_scalar(tag='episode_reward', value=episode_rewards[episode], step=episode)
        tf_logger.log_scalar(tag='steps_per_episode', value=steps_per_episode, step=episode)

        if solved:
            break
