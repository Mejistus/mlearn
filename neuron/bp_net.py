import numpy as np
import math
import numbers


class Neuron:
    def sigmoid(x: numbers.Number):
        return 1 / (1 + math.exp(-x))

    def liner(x: numbers.Number):
        return x

    def __init__(self, weight: float, bias: float, func=sigmoid, activation=0):
        self.w = weight
        self.b = bias
        self.fw_connextions = []
        self.bw_connextions = []
        self.func = func
        self.a = activation

    def add_connextions(self, neuron):
        self.fw_connextions.append(neuron)
        self.bw_connextions.append(neuron)

    def forward(self):
        self.a = self.func(self.a)
        for conn in self.fw_connextions:
            conn: Neuron
            conn.a += self.w * self.a + self.b

    def update(self):
        self.a = self.func(self.a)

    def backward(self):
        for conn in self.bw_connextions:
            conn: Neuron

    def __repr__(self) -> str:
        return f"Neuron[{self.w:.2f},{self.b:2f},{self.a:.2f}]"

    def __str__(self) -> str:
        return self.__repr__()


class layer:
    def __init__(self, neuron_num: int):
        self.neuron_num = neuron_num
        return

    def connect(self, layer):
        return self


class Train:
    def __init__(self, layers: list[layer]):
        return

    def fit(self, x: list[float], Y: list[float]):
        return

    def output():
        return


from tqdm import trange

i1, i2 = (
    Neuron(1, 0, activation=1, func=Neuron.liner),
    Neuron(1, 0, activation=0.5, func=Neuron.liner),
)
h11, h21 = (
    Neuron(0.9, 0),
    Neuron(0.3, 0),
)

o1, o2 = Neuron(0.2, 0), Neuron(0.8, 0)


input_layer = [i1, i2]
hidden_layer = [h11, h21]
output_layer = [o1, o2]

for i, neu in enumerate(input_layer):
    input_layer[i].add_connextions(hidden_layer[i])

for i, _ in enumerate(hidden_layer):
    for j, __ in enumerate(output_layer):
        hidden_layer[i].add_connextions(output_layer[j])


def train():
    # ? forward
    for i, _ in enumerate(input_layer):
        input_layer[i].forward()

    for i in input_layer:
        print(i, end=" ")
    print()
    for i, _ in enumerate(hidden_layer):
        hidden_layer[i].forward()

    for i in hidden_layer:
        print(i, end=" ")
    print()
    for i in output_layer:
        print(i, end=" ")
    # ?backward
    return


train()

# batches = 10000
# for _ in trange(batches):
#     pass
