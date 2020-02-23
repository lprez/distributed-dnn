from layer import *
from mip import *
from math import ceil

class Partition(Layer):
    def __init__(self, nodes, layer, lip, lop, fuse1, fuse2, previous_is_lop):
        self.lip = lip
        self.lop = lop
        self.fuse1 = fuse1
        self.fuse2 = fuse2
        self.previous_is_lop = previous_is_lop
        self.nodes = nodes
        self.layer = layer

    def multiplications(self):
        return ceil(self.layer.multiplications() // self.nodes)

    def weights_stored(self):
        return ceil(self.layer.weights_stored() // self.nodes)

    def input_size(self):
        return self.layer.input_size()

    def output_size(self):
        return self.layer.output_size()

    def communication_demand(self):
        return self.communication_demand_with(self.lip.x, self.lop.x, self.fuse1.x, self.fuse2.x, self.previous_is_lop.x * self.lop.x)

    def communication_demand_with(self, lip, lop, fuse1, fuse2, lop_product):
        i = self.input_size_total()
        o = self.output_size_total()
        n = self.nodes

        s = lip * (i * (n - 1) // n + o * (n - 1))
        s += lop_product * (- (i // n) * (n - 1)) + lop * (i * (n - 1) + o * (n - 1) // n)
        s += fuse1 * (i * (n - 1))
        s += fuse2 * (o * (n - 1))

        return s

    def __str__(self):
        if self.lop.x == 1:
            name = "LOP"
        elif self.lip.x == 1:
            name = "LIP"
        elif self.fuse1.x == 1:
            name = "FUSE1"
        elif self.fuse2.x == 1:
            name = "FUSE2"
        else:
            name = "???"
        return "[" + name + ":" + str(self.nodes) + "] " + str(self.layer) + " (mac: {:d}, mem: {:d}, bw: {:d})".format(
                    self.multiplications(), self.weights_stored(), self.communication_demand())

def optimize(root_layer, nodes):
    model = Model()
    zero_var = model.add_var(var_type = BINARY)
    model += zero_var == 0

    def partition(state, layer):
        nonlocal model
        cost, previous_is_lop, previous_is_fuse1 = state
        if isinstance(layer, Partitionable):
            lip = model.add_var(var_type = BINARY)
            lop = model.add_var(var_type = BINARY)
            fuse1 = model.add_var(var_type = BINARY)
            fuse2 = model.add_var(var_type = BINARY)
            lop_product = model.add_var(var_type = BINARY)
            new_layer = Partition(nodes, layer, lip, lop, fuse1, fuse2, previous_is_lop)
            model += lip + lop + fuse1 + fuse2 == 1
            model += fuse2 == previous_is_fuse1

            # previous_is_lop * lop == lop_product
            model += lop_product <= lop
            model += lop_product <= previous_is_lop
            model += lop_product >= lop + previous_is_lop - 1

            return (cost + new_layer.communication_demand_with(lip, lop, fuse1, fuse2, lop_product), lop, fuse1), new_layer
        elif isinstance(layer, Individual):
            return (cost, previous_is_lop, previous_is_fuse1), layer
        else:
            return (cost, zero_var, zero_var), layer


    (final_cost, last_is_lop, last_is_fuse1), new_layer = root_layer.map_accum(partition, (0, zero_var, zero_var))
    model += last_is_fuse1 == 0
    model.objective = minimize(final_cost)
    model.optimize()
    # return root_layer.communication_demand(), root_layer.multiplications(), new_layer.communication_demand(), new_layer.multiplications()
    return new_layer
