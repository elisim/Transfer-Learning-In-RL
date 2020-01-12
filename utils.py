from sklearn.preprocessing import StandardScaler
import numpy as np


class StateScaler:
    """
    scale and normalize the state (subtracts the mean and normalizes states to unit variance).
    """

    def __init__(self, env, n_samples=10000):
        state_samples = np.array([env.observation_space.sample() for _ in range(n_samples)])
        self.scaler = StandardScaler()
        self.scaler.fit(state_samples)

    def scale(self, state):
        """
        input shape = (2,)
        output shape = (1,2)
        """
        len_of_vec = state.shape[1]
        scaled = self.scaler.transform([state[0, 0:2].reshape([2])])
        new_state = np.pad(scaled.reshape(-1), (0, len_of_vec - 2))
        new_state = new_state.reshape([1, len_of_vec])

        return new_state
