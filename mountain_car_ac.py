# https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
import gym
import numpy as np
import tensorflow as tf
import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sklearn.preprocessing

tf.get_logger().setLevel(tf.logging.ERROR)
np.random.seed(1)
env = gym.make('MountainCarContinuous-v0')

retries = 0  # number of retries to run main
date_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]


class TensorFlowLogger:
    def __init__(self):
        self._log_dir = "logs/" + date_now
        self._file_writer = tf.summary.FileWriter(self._log_dir + "/metrics")

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._file_writer.add_summary(summary, step)


class StateScaler:
    """
    scale and normalize the state (subtracts the mean and normalizes states to unit variance).
    """
    def __init__(self, n_samples=10000):
        state_samples = np.array([env.observation_space.sample() for _ in range(n_samples)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(state_samples)

    def scale(self, state):
        """
        input shape = (2,)
        output shape = (1,2)
        """
        scaled = self.scaler.transform([state])
        return scaled


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


class PolicyNetwork:
    """
    The network input is the state and output are two scalar functions, μ(s) and σ(s),
    which are used as the mean and standard deviation of a Gaussian (normal) distribution.
    We will choose our actions by sampling from this distribution.
    """
    def __init__(self, tf_session, learning_rate):
        self.sess = tf_session
        self.learning_rate = learning_rate

        # placeholders to be fed
        self.state_placeholder = tf.placeholder(tf.float32, [None, 2])
        self.td_error_placeholder = tf.placeholder(tf.float32)
        self.action_placeholder = tf.placeholder(tf.float32)

        # build model
        # with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()

        # two hidden layers
        hidden1 = tf.layers.dense(self.state_placeholder, 32, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, 32, tf.nn.elu, init_xavier)

        # two outputs: mu and sigma
        mu = tf.layers.dense(hidden2, 1, None, init_xavier)
        sigma = tf.layers.dense(hidden2, 1, None, init_xavier)
        sigma = tf.nn.softplus(sigma) + 1e-5

        # create normal distribution and get action variable
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        action_tf_var = tf.clip_by_value(action_tf_var, env.action_space.low[0], env.action_space.high[0])
        self.action_tf_var = action_tf_var

        # define actor loss function and training operation
        self.loss = -tf.log(norm_dist.prob(self.action_placeholder) + 1e-5) * self.td_error_placeholder
        self.training_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def get_action(self, state):
        action = self.sess.run(self.action_tf_var, feed_dict={self.state_placeholder: state})
        return action

    def fit(self, action, state, td_error):
        # Update actor by minimizing loss
        self.sess.run([self.training_op, self.loss],
                      feed_dict={self.action_placeholder: action, self.state_placeholder: state,
                                 self.td_error_placeholder: td_error})

    def save_model(self):
        self.saver.save(sess=self.sess, save_path=f"models/policy_model.ckpt")

    def load_model(self, path):
        self.saver.restore(self.sess, save_path=path)


def main():
    with tf.Session() as sess:

        # Define hyperparameters
        max_episodes = 5000
        max_steps = 1000
        discount_factor = 0.99
        actor_learning_rate = 0.0004
        critic_learning_rate = 0.05

        # Create Logger to log scalars and StateScaler to scale to states
        tf_logger = TensorFlowLogger()
        state_scaler = StateScaler()

        # actor critic
        actor = PolicyNetwork(sess, actor_learning_rate)
        critic = CriticNetwork(critic_learning_rate)()
        sess.run(tf.global_variables_initializer())

        solved = False
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        total_steps = 0

        for episode in range(max_episodes):
            steps_per_episode = 1
            state = env.reset()

            # Generate an episode, out is list of episode_transitions: state, action, reward, next_state and done.
            for step in range(max_steps):
                # action.shape = (1,1)
                action = actor.get_action(state_scaler.scale(state))

                # Execute action and observe reward & next state from env
                # next_state shape=(2,)
                # env.step() requires input shape = (1,)
                next_state, reward, done, _ = env.step(np.squeeze(action, axis=0))
                episode_rewards[episode] += reward

                # V_of_next_state.shape=(1,1)
                V_of_next_state = critic.predict(state_scaler.scale(next_state))

                # Set TD Target: target = r + gamma * V(next_state)
                target = reward + discount_factor * np.squeeze(V_of_next_state)

                # td_error = target - V(s)
                V_of_curr_state = critic.predict(state_scaler.scale(state))
                td_error = target - np.squeeze(V_of_curr_state)

                # Update actor
                actor.fit(action=np.squeeze(action), state=state_scaler.scale(state), td_error=td_error)

                # Update critic
                critic.fit(state_scaler.scale(state), np.array(target, ndmin=2), verbose=0)

                if done:
                    # Check if stuck in local minima
                    if episode > 50 and np.mean(episode_rewards[(episode-49):(episode+1)]) < 0: # zero mean over last 50 episodes
                        global retries
                        retries += 1
                        return False

                    # Check if solved
                    if episode > 98:
                        average_rewards = np.mean(episode_rewards[(episode - 99):(episode + 1)])
                    if average_rewards > 90:  # Get a reward over 90
                        print('Solved at episode: ' + str(episode))
                        print('Total Steps: ' + str(total_steps))
                        critic.save_weights(f"models/critic_network.h5")
                        actor.save_model()
                        print('Saved Actor-Critic models')
                        solved = True
                    break

                state = next_state
                steps_per_episode += 1

            total_steps += steps_per_episode

            if solved:
                break

            print(f"Episode: {episode}, Number of Steps : {steps_per_episode}, Cumulative reward: {episode_rewards[episode]:0.2f}")
            tf_logger.log_scalar(tag='average_100_episodes_reward', value=average_rewards, step=episode)
            tf_logger.log_scalar(tag='episode_reward', value=episode_rewards[episode], step=episode)
            tf_logger.log_scalar(tag='steps_per_episode', value=steps_per_episode, step=episode)


if __name__ == '__main__':
    print("========== TRAIN START ==========")
    success = main()
    while not success:
        print(f"========== FAILED ==========")
        print(f"retries = {retries}")
        success = main()
    print(f"DONE. retries: {retries}")
