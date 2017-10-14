import sonnet as snt
import tensorflow as tf:

class QNetwork(snt.AbstractModule):

    def __init__(self, layer_sizes, layer_activations, output_activation):
        super(MLP, self).__init__(name='q_network')
        self._network = snt.nets.MLP(layer_sizes, activation=layer_activations)
        self._output_activation = output_activation

    def _build(self, input):
        net_out = self._network(input)
        return self._output_activation(net_out)
