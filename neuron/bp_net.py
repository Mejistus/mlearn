import random
import math
from tqdm import trange


class GenData:
    def __init__(self):
        self.animals = ["dog", "cow", "monkey"]

    def generate_data(self):
        data = []
        labels = []
        data = [
            # Dogs
            [1.2, 18.7, 30.5],
            [3.4, 45.2, 41.3],
            [5.6, 70.1, 50.2],
            [7.8, 95.5, 62.3],
            [10.0, 120.3, 70.4],
            [1.5, 22.1, 32.7],
            [2.5, 35.0, 38.4],
            [4.2, 55.6, 46.2],
            [6.3, 82.0, 58.3],
            [8.4, 105.5, 66.0],
            [1.8, 25.0, 33.5],
            [2.8, 38.4, 39.7],
            [4.8, 60.2, 48.5],
            [6.5, 85.0, 59.8],
            [9.5, 115.0, 68.0],
            [1.0, 15.0, 28.0],
            [3.0, 40.0, 42.0],
            [5.0, 65.0, 52.0],
            [7.0, 90.0, 60.0],
            [9.0, 115.0, 69.0],
            # Cows
            [1.2, 160.5, 110.5],
            [3.4, 400.0, 140.0],
            [5.6, 700.0, 160.0],
            [7.8, 950.0, 180.0],
            [10.0, 1200.0, 200.0],
            [2.0, 220.0, 120.0],
            [4.0, 480.0, 150.0],
            [6.0, 750.0, 170.0],
            [8.0, 1000.0, 190.0],
            [12.0, 1350.0, 210.0],
            [1.5, 180.0, 115.0],
            [3.5, 420.0, 145.0],
            [5.5, 680.0, 165.0],
            [7.5, 930.0, 185.0],
            [9.5, 1180.0, 205.0],
            [2.5, 250.0, 125.0],
            [4.5, 500.0, 155.0],
            [6.5, 780.0, 175.0],
            [8.5, 1050.0, 195.0],
            [11.5, 1400.0, 215.0],
            # Monkeys
            [1.2, 18.7, 50.5],
            [3.4, 30.2, 60.3],
            [5.6, 45.1, 70.2],
            [7.8, 60.5, 75.3],
            [10.0, 75.3, 80.4],
            [2.5, 22.1, 55.7],
            [4.5, 35.0, 65.4],
            [6.5, 50.6, 72.2],
            [8.5, 65.0, 78.3],
            [12.0, 80.0, 82.0],
            [3.0, 25.0, 58.5],
            [5.0, 38.4, 68.7],
            [7.0, 55.2, 74.5],
            [9.0, 70.0, 79.8],
            [11.0, 85.0, 83.0],
            [2.0, 20.0, 53.0],
            [4.0, 32.0, 63.0],
            [6.0, 48.0, 71.0],
            [8.0, 62.0, 77.0],
            [10.0, 77.0, 81.0],
        ]
        labels = [
            # Dogs
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            # Cows
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            # Monkeys
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
        # for _ in trange(100):
        #     animal = random.choice(self.animals)
        #     if animal == "dog":
        #         age = random.random() * 3 + 1
        #         weight = list(map(lambda x: x * (random.random() + 5) + 1, [age]))[0]
        #         height = list(map(lambda x: x * (random.random() * 2 + 8) + 10, [age]))[
        #             0
        #         ]
        #
        #         label = [1, 0, 0]
        #     elif animal == "cow":
        #         age = random.random() * 5 + 1
        #         weight = list(
        #             map(lambda x: x * (random.random() * 8 + 90) + 15, [age])
        #         )[0]
        #         height = list(
        #             map(lambda x: x * (random.random() * 5 + 25) + 45, [age])
        #         )[0]
        #         label = [0, 1, 0]
        #     elif animal == "monkey":
        #         age = random.random() * 8 + 1
        #         weight = list(map(lambda x: x * (random.random() * 1 + 2) + 25, [age]))[
        #             0
        #         ]
        #         height = list(map(lambda x: x * (random.random() * 1 + 3) + 45, [age]))[
        #             0
        #         ]
        #         label = [0, 0, 1]
        #
        #     data.append([age, weight, height])
        #     labels.append(label)

        return data, labels


import random
import math
from tqdm import trange


class MyNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input = input_size
        self.hidden = hidden_size
        self.output = output_size
        self.lr = learning_rate
        self.Wi2h = [
            [random.uniform(-1, 1) for _ in range(hidden_size)]
            for _ in trange(input_size)
        ]
        self.Wh2o = [
            [random.uniform(-1, 1) for _ in range(output_size)]
            for _ in trange(hidden_size)
        ]
        self.Bi2h = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.Bh2o = [random.uniform(-1, 1) for _ in range(output_size)]

    def func(self, x):
        if x < -25:
            return 0.000001
        if x > 10:
            return 0.9999
        return 1 / (1 + math.exp(-x))

    def Dfunc(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        # Input to hidden layer
        Oi2h = [
            sum(x * w for x, w in zip(inputs, col)) + b
            for col, b in zip(zip(*self.Wi2h), self.Bi2h)
        ]
        Oh = list(map(self.func, Oi2h))

        # Hidden to output layer
        Oh2o = [
            sum(x * w for x, w in zip(Oh, col)) + b
            for col, b in zip(zip(*self.Wh2o), self.Bh2o)
        ]
        Oo = list(map(self.func, Oh2o))
        return Oh, Oo

    def backward(self, inputs, hidden_outputs, actual_outputs, expected_outputs):
        # Output layer error and delta
        Eout = [
            expected - actual
            for expected, actual in zip(expected_outputs, actual_outputs)
        ]
        Dout = [
            error * self.Dfunc(output) for error, output in zip(Eout, actual_outputs)
        ]

        # Hidden layer error and delta
        Ehide = [
            sum(delta * w for delta, w in zip(Dout, col)) for col in zip(*self.Wh2o)
        ]
        Dhide = [
            error * self.Dfunc(output) for error, output in zip(Ehide, hidden_outputs)
        ]

        # Update weights and biases from hidden to output layer
        for i, hidden_output in enumerate(hidden_outputs):
            for j, output_delta in enumerate(Dout):
                self.Wh2o[i][j] += self.lr * output_delta * hidden_output
        for j, output_delta in enumerate(Dout):
            self.Bh2o[j] += self.lr * output_delta

        # Update weights and biases from input to hidden layer
        for i, input_val in enumerate(inputs):
            for j, hidden_delta in enumerate(Dhide):
                self.Wi2h[i][j] += self.lr * hidden_delta * input_val
        for j, hidden_delta in enumerate(Dhide):
            self.Bi2h[j] += self.lr * hidden_delta

    def train(self, training_data, training_labels, epochs):
        for epoch in trange(epochs):
            for inputs, expected_outputs in zip(training_data, training_labels):
                hidden_outputs, actual_outputs = self.forward(inputs)
                self.backward(inputs, hidden_outputs, actual_outputs, expected_outputs)

    def predict(self, inputs):
        Oh, Oo = self.forward(inputs)
        return Oo


def test(test_data, test_label):
    for i in range(len(test_data)):
        test, label = test_data[i], test_label[i]
        ans = nn.predict(test)
        output = g.animals[ans.index(max(ans))]
        print(
            "Predicted animal is",
            output,
            "should be",
            g.animals[label.index(1)],
            label.index(1) == ans.index(max(ans)),
        )


epochs = 50000
import itertools

g = GenData()
data, labels = g.generate_data()
data_set = list(zip(data, labels))
shff = random.sample(data_set, len(data_set))
data_set = shff
train_set = data_set[:50]
test_set = data_set[50:]

data, labels = list(zip(*train_set))
test_data, test_label = list(zip(*test_set))

nn = MyNN(3, 5, 3, 0.01)
nn.train(data, labels, epochs)
test(test_data, test_label)
