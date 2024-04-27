import math

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        if isinstance(other, Value):
          out = Value(self.data + other.data, (self, other), '+')
          def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0 
          out._backward = _backward
          return out
        else:
          other = Value(other)
          return self + other
    
    def __mul__(self, other):
        if isinstance(other, Value):
          out = Value(self.data * other.data, (self, other), '*')
          def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
          out._backward = _backward
          return out
        else:
          other = Value(other)
          return self * other

    def __rmul__(self, other):
      return self * other
    
    def exp(self):
      out = Value(math.exp(self.data), (self,), _op='e**')
      def _backward():
        self.grad += out.data * out.grad
      out._backward = _backward
      return out

    def __pow__(self, other):
      out = Value(self.data ** other, (self, ), _op=f'**{other}')
      def _backward():
        self.grad += out.grad * other * (self.data ** (other -1))
      out._backward = _backward
      return out
    
    def __truediv__(self, other):
      return self * other ** -1
    
    def __sub__(self, other):
      return self + (-other)
    
    def __neg__(self):
      return self * -1

    def __radd__(self, other):
      return self + other

    def tanh(self):
      x = self.data
      t = (math.exp(2*x) -1) / (math.exp(2*x) +1)
      out = Value(t, (self, ), _op='tanh')
      def _backward():
        self.grad += (1 - out.data **2) * out.grad
      out._backward = _backward
      return out

    def backward(self):
      visited = set()
      topo = []
      def build_topo(v):
        if v not in visited:
          visited.add(v)
          for child in v._prev:
            build_topo(child)
          topo.append(v)
      build_topo(self)
      self.grad = 1
      for node in reversed(topo):
        node._backward()