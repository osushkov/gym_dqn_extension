
import numpy as np

class MemoryChunk(object):

    def __init__(self, states, actions, rewards, next_states, is_terminal):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.is_terminal = is_terminal


class Memory(object):

    def __init__(self, max_capacity, state_shape):
        self._max_capacity = max_capacity
        self._cur_entries = 0

        state_shape = (max_capacity, ) + state_shape

        self._states = np.zeros(state_shape)
        self._actions = np.zeros((max_capacity, 1))
        self._rewards = np.zeros((max_capacity, 1))
        self._next_states = np.zeros(state_shape)
        self._is_terminal = np.zeros((max_capacity, 1), dtype=np.bool)

    def num_entries(self):
        return self._cur_entries

    def add_memory(self, state, action, reward, next_state):
        self._states[self._cur_entries] = state
        self._actions[self._cur_entries] = action
        self._rewards[self._cur_entries] = reward

        if next_state is None:
            self._is_terminal[self._cur_entries] = True
        else:
            self._is_terminal[self._cur_entries] = False
            self._next_states[self._cur_entries] = next_state

        self._cur_entries += 1

        if self._cur_entries >= self._max_capacity:
            self._purge_old_memories()

    def sample(self, num_samples):
        indices = np.random.randint(0, self._cur_entries, num_samples)
        return MemoryChunk(self._states[indices],
                           self._actions[indices],
                           self._rewards[indices],
                           self._next_states[indices],
                           self._is_terminal[indices])

    def _purge_old_memories(self):
        self._cur_entries = int(0.8 * self._cur_entries)

        self._states = self._states[-self._cur_entries:]
        self._actions = self._actions[-self._cur_entries:]
        self._rewards = self._rewards[-self._cur_entries:]
        self._next_states = self._next_states[-self._cur_entries:]
        self._is_terminal = self._is_terminal[-self._cur_entries:]
