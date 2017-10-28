import sonnet as snt
import tensorflow as tf

class QNetwork(snt.AbstractModule):

    def __init__(self, layer_sizes, num_actions, layer_activations, output_activation):
        super(QNetwork, self).__init__(name='q_network')

        self._network = snt.nets.MLP(layer_sizes, activation=layer_activations)
        self._value = snt.Linear(output_size=1)
        self._advantage = snt.Linear(output_size=num_actions)

        self._output_activation = output_activation

    def _build(self, input):
        flattened = snt.BatchFlatten()(input)
        net_out = self._network(flattened)
        value = self._output_activation(self._value(net_out))
        advantage = self._output_activation(self._advantage(net_out))

        mean_advantage = tf.reshape(tf.reduce_mean(advantage, 1), (-1, 1))
        return value + (advantage - mean_advantage)

    def get_variables(self, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
        return (self._network.get_variables(collection) +
                self._value.get_variables(collection) +
                self._advantage.get_variables(collection))
