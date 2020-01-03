import gym
import numpy as np
import tensorflow as tf
import collections
import datetime
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('Acrobot-v1')
np.random.seed(1)


class TensorFlowLogger:
    def __init__(self, with_baseline):
        self._log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if with_baseline:
            self._log_dir += "_with_baseline"
        self._file_writer = tf.summary.FileWriter(self._log_dir + "/metrics")

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._file_writer.add_summary(summary, step)


class ActorCritic:
    def __init__(self, state_size, action_size, actor_learning_rate, critic_learning_rate, discount_factor, sess, tf_logger):
        self.state_size = state_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.sess = sess
        self.tf_logger = tf_logger

        self.actor = PolicyNetwork(self.state_size, self.action_size, self.actor_learning_rate)
        self.critic = CriticNetwork(self.state_size, self.critic_learning_rate)()
        self.saver = tf.train.Saver()

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

    def save_model(self, file_name):
        self.critic.save_weights(f'models/critic/{file_name}.h5')
        self.saver.save(sess=self.sess, save_path=f'models/actor/{file_name}.ckpt')

    def load_model(self, file_name):
        self.critic.save_weights(f'models/critic/{file_name}.h5')
        self.saver.restore(self.sess, save_path=f'models/actor/{file_name}.ckpt')


class CriticNetwork:
    def __init__(self, state_size, lerning_rate):
        self.state_size = state_size
        self.learning_rate = lerning_rate

        self.critic = Sequential()
        self.critic.add(Dense(units=32, input_dim=self.state_size, activation='relu'))
        self.critic.add(Dense(units=32, activation='relu'))
        self.critic.add(Dense(units=32, activation='relu'))
        self.critic.add(Dense(units=1))
        self.critic.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def __call__(self):
        return self.critic


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):  # define network variables

            # Inserts a placeholder for a tensor that will be always fed.
            # Its value must be fed using the feed_dict optional argument to Session.run()
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            # two Dense layers
            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)  # Z1 = State * W1 +  b1
            self.A1 = tf.nn.relu(self.Z1)  # A1 = relu(Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)  # output = A1 * W2 + b2 (logits)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))  # actions probs = softmax(output)
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,
                                                                           labels=self.action)  # cross-entropy with actions probs.
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def main():
    run_with_baseline_flag = "-b" in sys.argv
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Define hyperparameters
    max_episodes = 500
    max_steps = 1000
    discount_factor = 0.99
    policy_learning_rate = 0.004
    critic_learning_rate = 0.0004
    render = False
    save_model = True
    load_model = False
    load_model_file_name = 'acrobot_model'
    save_model_file_name = 'acrobot_model'


    # Create Logger to log scalars
    tf_logger = TensorFlowLogger(with_baseline=run_with_baseline_flag)

    # Initialize the policy network
    tf.reset_default_graph()

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        model = ActorCritic(state_size, action_size, policy_learning_rate, critic_learning_rate, discount_factor, sess, tf_logger)
        if load_model:
            model.load_model(load_model_file_name)
        sess.run(tf.global_variables_initializer())
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
                    break
                state = next_state
                steps_per_episode += 1

            tf_logger.log_scalar(tag='average_100_episodes_reward', value=average_rewards, step=episode)
            tf_logger.log_scalar(tag='episode_reward', value=episode_rewards[episode], step=episode)
            tf_logger.log_scalar(tag='steps_per_episode', value=steps_per_episode, step=episode)

        if save_model:
            model.save_model(save_model_file_name)


if __name__ == '__main__':
    main()
