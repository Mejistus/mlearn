import itertools
import functools
from math import log2
from collections import defaultdict
from pprint import pprint


def load_data(filename="./周志华_西瓜数据集2.txt"):
    data = []
    with open(filename, "r") as f:
        while f.readable() and (line := f.readline()):
            data.append(line.strip().split(","))
    return data


class DT:

    def __init__(self, is_leaf=None, **kwargs):
        self.next = defaultdict()
        self.criterion = None
        self.data: list = []
        self.attr: list = []

        for key, val in kwargs.items():
            match key:
                case "data":
                    self.data = kwargs[key]
                case "attr":
                    self.attr = kwargs[key]

    def select_by_attr(self, **kwargs):
        """select all row if this row meets kwargs"""

        def comp(row):
            for key, val in kwargs.items():
                if row[self.attr.index(key)] != val:
                    return False
            return True

        li = filter(lambda i: comp(i), self.data)
        return list(li)

    def max_entropy(self, positive):
        """only for columns >=2"""
        order = []
        for i, cur_attr in enumerate(self.attr[:-1]):
            fields = set([i[self.attr.index(cur_attr)] for i in self.data])
            ent = 0
            for attv in fields:
                selected = self.select_by_attr(**{cur_attr: attv})
                posi = self.select_by_attr(
                    **{self.attr[-1]: positive, cur_attr: attv},
                )

                k, m, n = len(posi), len(selected), len(self.data)
                if not (k == 0 or k == m):
                    ent += (k / m * log2(k / m) + (1 - k / m)
                            * log2(1 - k / m)) * m / n
            order.append([i, -ent])
        order.sort(key=lambda x: x[-1])
        return order[0]

    def build(self):
        selected_attr_id, ent = self.max_entropy(positive="是")
        fields = set([i[selected_attr_id] for i in self.data])
        node = DT()
        node.data = self.data
        node.attr = self.attr
        node.attr = node.attr[:selected_attr_id] + \
            node.attr[selected_attr_id + 1:]
        node.data = list(
            map(
                lambda x: x[:selected_attr_id] + x[selected_attr_id + 1:],
                node.data,
            )
        )

        if len(self.attr) == 2:
            for attv in fields:
                node.v = attv
                selected = self.select_by_attr(
                    **{self.attr[selected_attr_id]: attv},
                )
                positive = self.select_by_attr(
                    **{self.attr[-1]: "是", self.attr[selected_attr_id]: attv},
                )
                k, m = len(positive), len(selected)
                if k / m > 0.7:
                    node.next[attv] = True
                elif k / m < 0.3:
                    node.next[attv] = False
                else:
                    pass
            return node

        for attr_value in fields:
            self.criterion = self.attr[selected_attr_id]
            self.next[attr_value] = node.build()
        return self

    def dfs(self, dpt=0):
        for i, j in self.next.items():
            print(f"{"  "*dpt}{self.criterion}={i}")
            if isinstance(j, DT):
                j.dfs(dpt + 1)


if __name__ == "__main__":
    data = load_data()
    attr, data = data[0], data[1:]
    dt = DT(data=data, attr=attr)
    dt = dt.build()
    dt.dfs()
