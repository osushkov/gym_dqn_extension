
import agent
import memory
import q_network

import math
import random
import numpy as np
import tensorflow as tf

from gym.spaces.discrete import Discrete

_LAYER_SIZES = (32,16)
_LAYER_ACTIVATION = tf.nn.elu
_OUTPUT_ACTIVATION = tf.identity
_TARGET_UPDATE_RATE = 5000
_LEARN_BATCH_SIZE = 64
_DISCOUNT = 0.98


class DQNLearner(agent.Agent):

    def __init__(self, action_space, observation_space, exploration_rate, memory):
        assert isinstance(action_space, Discrete)

        self._action_space = action_space
        self._observation_space = observation_space
        self._exploration_rate = exploration_rate

        self._learning_flag = True
        self._state_shape = observation_space.high.shape
        self._memory = memory
        self._cur_exploration = self._exploration_rate(0)

        self._last_action = None
        self._last_state = None

        self._learn_iters_since_update = 0

        self._build_graph()

        self._sess = tf.Session(graph=self._graph)
        with self._sess.as_default():
            self._sess.run(self._init_op)
            self._sess.run(self._target_update_ops)

    def initialize_episode(self, episode_count):
        self._cur_exploration = self._exploration_rate(episode_count)
        self._memory.initialize_episode(episode_count)

    def act(self, observation):
        observation = self._normalised_state(observation)

        if self._learning_flag:
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
        reward /= 50.0

        if episode_done:
            resulting_state = None

        self._memory.add_memory(self._last_state, self._last_action, reward, resulting_state)

    def set_learning(self, learning_flag):
        self._learning_flag = learning_flag

    def _random_action(self):
        return self._action_space.sample()

    def _best_action(self, observation):
        return np.argmax(self._q_values(observation))

    def _q_values(self, observation):
        observation = observation.reshape((1,) + self._state_shape)
        with self._sess.as_default():
            return self._sess.run(self._q_output, feed_dict={self._q_observation: observation})

    def _learn(self):
        if self._memory.num_entries() < self._memory.capacity() / 10:
            return

        mem_chunk = self._memory.sample(_LEARN_BATCH_SIZE)
        feed_dict = {
                self._weights : mem_chunk.weights,
                self._learn_observation : mem_chunk.states,
                self._learn_action : mem_chunk.actions,
                self._reward : mem_chunk.rewards,
                self._target_observation : mem_chunk.next_states,
                self._learn_next_state: mem_chunk.next_states,
                self._target_is_terminal : mem_chunk.is_terminal,
        }

        self._learn_iters_since_update += 1
        with self._sess.as_default():
            _, loss, td_error = self._sess.run((self._learn_optimizer, self._learn_loss,
                                                self._td_error), feed_dict=feed_dict)

            self._memory.update_p_choice(td_error)

            if self._learn_iters_since_update >= _TARGET_UPDATE_RATE:
                self._sess.run(self._target_update_ops)
                self._learn_iters_since_update = 0

    def _build_graph(self):
        self._graph = tf.Graph()

        with self._graph.as_default():
            self._build_q_network()
            self._build_target_network()
            self._build_learn_loss()
            self._build_update_ops()

            self._init_op = tf.global_variables_initializer()

    def _build_q_network(self):
        self._q_observation = tf.placeholder(tf.float32, shape=((1, ) + self._state_shape))

        self._q_network = q_network.QNetwork(_LAYER_SIZES, self._action_space.n,
                                             _LAYER_ACTIVATION, _OUTPUT_ACTIVATION)

        self._q_output = self._q_network(self._q_observation)

    def _build_target_network(self):
        self._target_observation = tf.placeholder(
                tf.float32, shape=((_LEARN_BATCH_SIZE, ) + self._state_shape))

        self._target_network = q_network.QNetwork(_LAYER_SIZES, self._action_space.n,
                                                  _LAYER_ACTIVATION, _OUTPUT_ACTIVATION)

        self._target_output = self._target_network(self._target_observation)

    def _build_learn_loss(self):
        self._weights = tf.placeholder(tf.float32, shape=_LEARN_BATCH_SIZE)
        self._learn_observation = tf.placeholder(
                tf.float32, shape=((_LEARN_BATCH_SIZE, ) + self._state_shape))
        self._learn_next_state = tf.placeholder(
                tf.float32, shape=((_LEARN_BATCH_SIZE, ) + self._state_shape))
        self._learn_action = tf.placeholder(tf.int32, shape=_LEARN_BATCH_SIZE)
        self._reward = tf.placeholder(tf.float32, shape=_LEARN_BATCH_SIZE)
        self._target_is_terminal = tf.placeholder(tf.bool, shape=_LEARN_BATCH_SIZE)

        index_range = tf.constant(np.arange(_LEARN_BATCH_SIZE), dtype=tf.int32)
        learn_output = self._q_network(self._learn_observation)

        best_actions = tf.argmax(self._q_network(self._learn_next_state), axis=1,
                                 output_type=tf.int32)
        best_action_indices = tf.stack([index_range, best_actions], axis=1)

        terminating_target = self._reward
        intermediate_target = self._reward + (tf.gather_nd(self._target_output, best_action_indices) * _DISCOUNT)
        desired_output = tf.stop_gradient(
            tf.where(self._target_is_terminal, terminating_target, intermediate_target))

        action_indices = tf.stack([index_range, self._learn_action], axis=1)
        indexed_output = tf.gather_nd(learn_output, action_indices)

        self._td_error = desired_output - indexed_output
        self._learn_loss = tf.losses.mean_squared_error(desired_output, indexed_output,
                                                        weights=self._weights)

        opt = tf.train.AdamOptimizer(0.0001)
        self._learn_optimizer = opt.minimize(self._learn_loss,
                                             var_list=self._q_network.get_variables())

    def _build_update_ops(self):
        qvars = self._q_network.get_variables()
        tvars = self._target_network.get_variables()

        self._target_update_ops = []
        for qvar, tvar in zip(qvars, tvars):
            self._target_update_ops.append(
                    tf.assign(tvar, qvar, validate_shape=True, use_locking=True))

    def _normalised_state(self, obs):
        # obs[0] /= self._observation_space.high[0] / 2.0
        # obs[1] /= 1.5
        # obs[2] /= self._observation_space.high[2] / 2.0
        # obs[3] /= 1.5
        return obs
        # obs_range = (self._observation_space.high - self._observation_space.low)
        # return (obs - self._observation_space.low) / obs_range
