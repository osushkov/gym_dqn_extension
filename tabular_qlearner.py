
import agent
import math
import numpy as np
from gym.spaces.discrete import Discrete

class TabularQLearner(agent.Agent):

    def __init__(self, action_space, observation_space, total_episodes, discount=0.99,
                 init_learn_rate=0.1, final_learn_rate=0.01,
                 init_egreedy=1.0, final_egreedy=0.1, space_buckets=50):

        self._learning_flag = True

        self._action_space = action_space
        self._observation_space = observation_space
        self._discount = discount

        self._init_learn_rate = init_learn_rate
        self._learn_rate = self._init_learn_rate
        self._learn_rate_decay = math.pow(final_learn_rate / init_learn_rate, 1.0 / total_episodes)

        self._init_egreedy = init_egreedy
        self._egreedy = self._init_egreedy
        self._egreedy_decay = math.pow(final_egreedy / init_egreedy, 1.0 / total_episodes)

        self._bucket_size = (observation_space.high - observation_space.low) / space_buckets

        q_shape = (space_buckets,)*self._bucket_size.shape[0] + (action_space.n,)
        self._q_table = np.random.normal(0.0, 0.1, q_shape)

        self._last_action = None
        self._last_state = None

        if not isinstance(action_space, Discrete):
            print("not discrete action space")

    def initialize_episode(self, episode_count):
        self._egreedy = self._init_egreedy * (self._egreedy_decay ** episode_count)
        self._learn_rate = self._init_learn_rate * (self._learn_rate_decay ** episode_count)

    def act(self, observation):
        if np.random.rand() < self._egreedy and self._learning_flag:
            action = self._random_action()
        else:
            action = self._best_action(observation)

        self._last_action = action
        self._last_state = observation

        return action

    def feedback(self, resulting_state, reward, episode_done):
        if episode_done:
            target = reward
        else:
            index = self._bucket_index(resulting_state)
            target = reward + self._discount * np.max(self._q_table[index])

        self._update_q_table(self._last_state, self._last_action, target)

    def set_learning(self, learning_flag):
        self._learning_flag = learning_flag

    def _random_action(self):
        return self._action_space.sample()

    def _best_action(self, observation):
        index = self._bucket_index(observation)
        return np.argmax(self._q_table[index])

    def _bucket_index(self, observation):
        index = ((observation - self._observation_space.low) / self._bucket_size).astype(int)
        return tuple(index)

    def _q_value(self, observation, action):
        return self._q_table[self._bucket_index(observation)][action]

    def _update_q_table(self, observation, action, target_value):
        shift = target_value - self._q_value(observation, action)
        shift *= self._learn_rate
        self._q_table[self._bucket_index(observation)][action] += shift
