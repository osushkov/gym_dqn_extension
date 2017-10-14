
import agent
import memory
import q_network

import math
import numpy as np
import tensorflow as tf

from gym.spaces.discrete import Discrete

_LAYER_SIZES = (16, 16)
_LAYER_ACTIVATION = tf.nn.tanh
_OUTPUT_ACTIVATION = tf.identity
_TARGET_UPDATE_RATE = 1000
_MEMORY_SIZE = 10000
_LEARN_BATCH_SIZE = 64
_DISCOUNT = 0.99


class DQNLearner(agent.Agent):

    def __init__(self, action_space, observation_space, exploration_rate):
        assert isinstance(action_space, Discrete)


        self._action_space = action_space
        self._observation_space = observation_space
        self._exploration_rate = exploration_rate

        self._learning_flag = True
        self._state_shape = observation_space.high.shape
        self._memory = memory.Memory(_MEMORY_SIZE, self._state_shape)
        self._cur_exploration = self._exploration_rate(0)

        self._last_action = None
        self._last_state = None

        self._learn_iters_since_update = 0
        self._build_graph()


    def initialize_episode(self, episode_count):
        self._cur_exploration = self._exploration_rate(episode_count)

    def act(self, observation):
        observation = self._normalised_state(observation)

        self._learning_flag:
            self._learn()

        if self._learning_flag and np.random.rand() < self._cur_exploration:
            action = self._random_action()
        else:
            action = self._best_action(observation)

        self._last_state = observation
        self._last_action = action

        return action

    def feedback(self, resulting_state, reward, episode_done):
        resulting_state = self._normalised_state(resulting_state)

        memory.add_memory(self._last_state, self._last_action, reward, resulting_state,
                          episode_done)

    def set_learning(self, learning_flag):
        self._learning_flag = learning_flag

    def _random_action(self):
        return self._action_space.sample()

    def _best_action(self, observation):
        return np.argmax(self._q_values(observation))

    def _q_values(self, observation):
        observation = observation.reshape((1,) + self._state_shape)
        with self._sess.as_default():
            return sess.run(self._q_output, feed_dict={self._q_observation: observation})

    def _build_graph(self):
        self._graph = tf.Graph()

        self._q_network = q_network.QNetwork(_LAYER_SIZES, _LAYER_ACTIVATION, _OUTPUT_ACTIVATION)
        self._target_network = q_network.QNetwork(
                _LAYER_SIZES, _LAYER_ACTIVATION, _OUTPUT_ACTIVATION)

        self._build_update_ops()

        self._q_observation = tf.placeholder(tf.float32, shape=((1, ) + self._state_shape))
        self._q_output = self._q_network(self._q_observation)

        self._learn_observation = tf.placeholder(
                tf.float32, shape=((_LEARN_BATCH_SIZE, ) + self._state_shape))
        self._learn_action = tf.placeholder(tf.int32, shape=(_LEARN_BATCH_SIZE, 1))
        self._reward = tf.placeholder(tf.float32, shape=(_LEARN_BATCH_SIZE, 1))
        self._target_observation = tf.placeholder(
                tf.float32, shape=((_LEARN_BATCH_SIZE, ) + self._state_shape))
        self._target_is_terminal = tf.placeholder(tf.bool, shape=(_LEARN_BATCH_SIZE, 1))

        learn_output = self._q_network(self._learn_observation)
        target_output = self._target_network(self._target_observation)

        terminating_target = self._reward
        intermediate_target = self._reward + (tf.reduce_max(target_output, axis=1) * _DISCOUNT)
        desired_output = tf.stop_gradient(
            tf.where(self._target_is_terminal, terminating_target, intermediate_target))
        indexed_output = tf.gather_nd(self.learn_output, self._learn_action)

        self._learn_loss = tf.losses.mean_squared_error(desired_output, indexed_output)

        opt = tf.train.AdamOptimizer()
        self._learn_optimizer = opt.minimize(self._learn_loss,
                                             var_list=self._q_network.get_variables())

        self._sess = tf.Session(graph=self._graph)
        with self._sess.as_default():
            self._sess.run(tf.global_variables_initializer())
            self._sess.run(self._target_update_ops)

    def _build_update_ops(self):
        qvars = self._q_network.get_variables()
        tvars = self._target_network.get_variables()

        self._taget_update_ops = []
        for qvar, tvar in zip(qvars, tvars):
            self._target_update_ops.append(
                    tf.assign(tvar, qvar, validate_shape=True, use_locking=True))

    def _learn(self):
        if self._memory.num_entries() < _MEMORY_SIZE / 10:
            return

        self._learn_iters_since_update += 1
        with self._sess.as_default():
            _, loss = self._sess.run((self._learn_optimizer, self._learn_loss))

            if self._learn_iters_since_update >= _TARGET_UPDATE_RATE:
                self._sess.run(self._target_update_ops)
                self._learn_iters_since_update = 0


    def _normalised_state(self, obs):
        return (obs - observation_space.low) / (observation_space.high - observation_space.low)
