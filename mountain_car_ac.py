# https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
import gym
import numpy as np
import tensorflow as tf
import collections
import datetime

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation
from keras.optimizers import Adam

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


class ActorCritic:
    def __init__(self, 
                 state_size, 
                 action_size, 
                 actor_learning_rate, 
                 critic_learning_rate, 
                 discount_factor, 
                 sess, 
                 tf_logger):
        self.state_size = state_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.sess = sess
        self.tf_logger = tf_logger

        self.actor = PolicyNetwork(self.state_size, self.action_size, self.actor_learning_rate)
        self.critic = CriticNetwork(self.state_size, self.critic_learning_rate)()

    def train(self, state, action, action_one_hot, reward, next_state, done, total_steps):
        target = np.zeros((1, 1))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * next_value - value
            target[0][0] = reward + self.discount_factor * next_value

        feed_dict = {self.actor.state: state,
                     self.actor.R_t: advantages,
                     self.actor.action: action_one_hot}
        _, loss = self.sess.run([self.actor.optimizer, self.actor.loss], feed_dict)
        self.tf_logger.log_scalar(tag='policy_network_loss', value=loss, step=total_steps)

        # self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)


class CriticNetwork:
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
    def __init__(self, learning_rate, name='policy_network'):
        self.learning_rate = learning_rate

        inp = Input(state_size)
        x = Dense(units=32, activation='elu')(inp)
        x = Dense(units=32, activation='elu')(x)
        sigma = Dense(units=1, activation='softplus')(x) + 1e-5  # σ
        mu = Dense(units=1)(x)  # μ
        model = Model(int, [sigma, mu])
        tf.random.Normal


def policy_network(state):
    n_hidden1 = 40
    n_hidden2 = 40
    n_outputs = 1

    with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()

        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier)
        mu = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.nn.softplus(sigma) + 1e-5
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        action_tf_var = tf.clip_by_value(
            action_tf_var, env.action_space.low[0],
            env.action_space.high[0])
    return action_tf_var, norm_dist

loss_actor = -tf.log(norm_dist.prob(action_placeholder) + 1e-5) * delta_placeholder
        training_op_actor = tf.train.AdamOptimizer(
            lr_actor, name='actor_optimizer').minimize(loss_actor)


def main():
#     run_with_baseline_flag = "-b" in sys.argv
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # Define hyperparameters
    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    policy_learning_rate = 0.0004
    critic_learning_rate = 0.0004
    render = False

    # Create Logger to log scalars
    tf_logger = TensorFlowLogger(with_baseline=run_with_baseline_flag)

    # Initialize the policy network
    tf.reset_default_graph()

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        model = ActorCritic(state_size, action_size, policy_learning_rate, critic_learning_rate, discount_factor, sess, tf_logger)
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        total_steps = 0

        for episode in range(max_episodes):
            steps_per_episode = 0
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []

            # Generate an episode, out is list of episode_transitions: state, action, reward, next_state and done.
            for step in range(max_steps):
                actions_distribution = sess.run(model.actor.actions_distribution, feed_dict={model.actor.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(
                    Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                model.train(state, action, action_one_hot, reward, next_state, done, total_steps)

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    print(
                        "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                     round(average_rewards, 2)))
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