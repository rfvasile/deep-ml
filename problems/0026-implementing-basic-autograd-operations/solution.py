class Value:
    def __init__(self, data, _children=(), _op="", _label="undefined"):
        self.data = data
        self.grad = 0
        self._prev = _children
        self._op = _op
        self._label = _label

    def __repr__(self):
        def fmt(x):
            return int(x) if float(x).is_integer() else round(x, 4)

        return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

    def __add__(self, other):
        data = self.data + other.data
        return Value(data, (self, other), "+", _label="add")

    def __mul__(self, other):
        data = self.data * other.data
        return Value(data, (self, other), "*", _label="mult")

    def relu(self):
        return Value(max(0, self.data), (self,), "relu", _label="relu")

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v.propagate()

    def propagate(self):
        if not self._prev:
            return

        if self._op == "relu":
            c1 = self._prev[0]

            # dRelu * self.grad
            c1.grad += (self.data > 0) * self.grad
        elif self._op == "+":
            c2 = self._prev[1]
            c1 = self._prev[0]

            # dPlus * self.grad
            c2.grad += 1 * self.grad
            c1.grad += 1 * self.grad
        elif self._op == "*":
            c2 = self._prev[1]
            c1 = self._prev[0]

            # dTimes * self.grad
            c2.grad += c1.data * self.grad
            c1.grad += c2.data * self.grad
        else:
            raise ValueError()
