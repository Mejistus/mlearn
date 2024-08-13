class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def backward(self):

        if self.grad is None:
            self.grad = 1.0

        stack = [self]
        while stack:
            t = stack.pop()
            t._backward()
            stack.extend(t._prev)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = (
                    other.grad + out.grad if other.grad is not None else out.grad
                )

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad = (
                    self.grad + other.data * out.grad
                    if self.grad is not None
                    else other.data * out.grad
                )
            if other.requires_grad:
                other.grad = (
                    other.grad + self.data * out.grad
                    if other.grad is not None
                    else self.data * out.grad
                )

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"


a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = a + b + a * b
c.backward()
print(c.grad)
print(a.grad)
print(b.grad)
