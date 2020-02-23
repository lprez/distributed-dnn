from math import floor

class Layer():
    # multiplications per node
    def multiplications(self):
        return NotImplemented

    def input_size(self):
        return NotImplemented

    def set_input_size(self, input_size):
        return NotImplemented

    def input_size_total(self):
        size = 1
        for s in self.input_size():
            size *= s
        return size

    def output_size(self):
        return NotImplemented

    def output_size_total(self):
        size = 1
        for s in self.output_size():
            size *= s
        return size

    def weights_stored(self):
        return NotImplemented

    def communication_demand(self):
        return NotImplemented

    def map_accum(self, func, init):
        return func(init, self)

class Partitionable(Layer):
    pass

class Individual(Layer):
    pass

class FC(Partitionable):
    def __init__(self, outputs, input_size = None):
        if input_size is not None:
            self.inputs = 1
            for s in input_size:
                self.inputs *= s
        else:
            self.inputs = None
        self.outputs = outputs

    def multiplications(self):
        return self.inputs * self.outputs

    def weights_stored(self):
        return self.inputs * self.outputs

    def input_size(self):
        return [self.inputs] if self.inputs is not None else None

    def output_size(self):
        return [self.outputs]

    def set_input_size(self, input_size):
        self.inputs = 1
        for s in input_size:
            self.inputs *= s

    def communication_demand(self):
        return 0

    def __str__(self):
        return 'FC {:d} -> {:d} (mac: {:d}, mem: {:d}, bw: {:d})'.format(
                    self.inputs, self.outputs, self.multiplications(), self.weights_stored(), self.communication_demand())

class Conv2d(Partitionable):
    def __init__(self, output_channels, kernel_size = [3, 3], stride = 1, padding = 0, input_size = None):
        self.inputs = input_size
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def multiplications(self):
        return self.input_size_total() * self.kernel_size[0] * self.kernel_size[1] * self.output_channels

    def weights_stored(self):
        return self.kernel_size[0] * self.kernel_size[1] * self.inputs[0] * self.output_channels

    def input_size(self):
        return self.inputs

    def set_input_size(self, input_size):
        self.inputs = input_size

    def output_size(self):
        h = floor((self.inputs[1] + 2 * self.padding - self.kernel_size[0]) / self.stride + 1)
        w = floor((self.inputs[2] + 2 * self.padding - self.kernel_size[1]) / self.stride + 1)
        return [self.output_channels, h, w]

    def communication_demand(self):
        return 0

    def __str__(self):
        return 'Conv2d {:d}x{:d} {:d} -> {:d} (mac: {:d}, mem: {:d}, bw: {:d})'.format(
                    self.kernel_size[0], self.kernel_size[1], self.inputs[0], self.output_channels,
                    self.multiplications(), self.weights_stored(), self.communication_demand())

class Pool2d(Individual):
    def __init__(self, channels, kernel_size = [3, 3], stride = 1, padding = 0, input_size = None):
        self.inputs = input_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def multiplications(self):
        return 0

    def weights_stored(self):
        return 0

    def input_size(self):
        return self.inputs

    def set_input_size(self, input_size):
        self.inputs = input_size

    def output_size(self):
        h = floor((self.inputs[1] + 2 * self.padding - self.kernel_size[0]) / self.stride + 1)
        w = floor((self.inputs[2] + 2 * self.padding - self.kernel_size[1]) / self.stride + 1)
        return [self.channels, h, w]

    def communication_demand(self):
        return 0

    def __str__(self):
        return 'Pool2d {:d}x{:d} {:d} -> {:d} (mac: {:d}, mem: {:d}, bw: {:d})'.format(
                    self.kernel_size[0], self.kernel_size[1], self.inputs[0], self.channels,
                    self.multiplications(), self.weights_stored(), self.communication_demand())

class Flatten(Layer):
    def __init__(self, input_size = None):
        self.inputs = input_size
        if input_size is not None:
            self.outputs = 1
            for os in input_size:
                self.outputs *= os
        else:
            self.outputs = None

    def multiplications(self):
        return 0

    def weights_stored(self):
        return 0

    def input_size(self):
        return self.inputs

    def set_input_size(self, input_size):
        self.inputs = input_size
        self.outputs = 1
        for os in input_size:
            self.outputs *= os

    def output_size(self):
        return [self.outputs]

    def communication_demand(self):
        return 0

    def __str__(self):
        return 'Flatten {:d}'.format(self.outputs)

class Sequential(Layer):
    def __init__(self, *layers):
        if not Sequential.check_size(layers):
            raise Exception("invalid input/output size")
        self.layers = layers

    def multiplications(self):
        n = 0
        for layer in self.layers:
            n += layer.multiplications()
        return n

    def weights_stored(self):
        n = 0
        for layer in self.layers:
            n += layer.weights_stored()
        return n

    def communication_demand(self):
        n = 0
        for layer in self.layers:
            n += layer.communication_demand()
        return n

    def input_size(self):
        return self.layers[0].input_size()

    def set_input_size(self, input_size):
        return self.layers[0].set_input_size(input_size)

    def output_size(self):
        return self.layers[-1].output_size()

    def check_size(layers):
        if layers[0].input_size() is None:
            return False

        valid = True
        for i in range(len(layers) - 1):
            #print(layers[i].output_size())
            #print(layers[i + 1].input_size())
            if layers[i + 1].input_size() is None:
                layers[i + 1].set_input_size(layers[i].output_size())
            elif Sequential.squeeze(layers[i].output_size()) != Sequential.squeeze(layers[i + 1].input_size()):
                valid = False
        return valid

    def map_accum(self, func, init):
        state = init
        new_layers = []
        for layer in self.layers:
            state, new_layer = layer.map_accum(func, state)
            new_layers.append(new_layer)
        return state, Sequential(*new_layers)

    def squeeze(size):
        new_size = []
        for s in size:
            if s > 1:
                new_size.append(s)
        return new_size

    def __str__(self):
        summary = 'MAC operations per device: {:d}, Memory footprint per device: {:d}, Total data exchanged: {:d}'.format(
                    self.multiplications(), self.weights_stored(), self.communication_demand())
        return "\n".join([str(l) for l in self.layers] + [summary])

class FCSequential(Sequential):
    def __init__(self, inputs, *outputs):
        layers = []
        for output in outputs:
            layers.append(FC(outputs, inputs))
            inputs = output
        super().__init__(*layers)

    def __str__(self):
        return Sequential.__str__(self)

class Conv2dSequential(Sequential):
    def __init__(self, input_size, *params):
        layers = []
        for ps in params:
            output_channels = ps[0]
            kernel_size = ps[1]
            stride = 1 if len(ps) < 3 else ps[2]
            padding = 0 if len(ps) < 4 else ps[3]
            layer = Conv2d(output_channels, kernel_size, stride, padding, input_size)
            layers.append(layer)
            input_size = layer.output_size()
        super().__init__(*layers)

    def __str__(self):
        return Sequential.__str__(self)
