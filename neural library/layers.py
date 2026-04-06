"""
neural nets are made up of layers
each layers needs to pass inputs forward
and propagate gradients forward

ex) inputs -> linear -> Tanh -> linear -> output
"""

from tensor import Tensor
import numpy as np
from typing import Dict, Callable

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        product outputs corresponding to the inputs
        """
        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        backpropagate this gradient through the layer
        """
        raise NotImplementedError
    

class Linear(Layer):
    # computes output = input @ w + b
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs (batch, input)
        # outputs (batch, output)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = ab + c
        then dy/da = f'(x) *b 
        and dy/db = f(x) * a
        and dy/dc = f' (x)
        layers.py
        if y = f(x) and x= a @b+ c
        then dy/da = f' (x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f' (x)
        """
        self.grads["b"] = np.sum(grad, axis = 0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    # applies a function element wise to its inputs
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_primt = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        # if y = f(x) and x = g(z)
        # then dy/dz = f'(x) * g'(z)
        return grad * self.f_prime(self.inputs)


    def tanh(x: Tensor) -> Tensor:
        return np.tanh(x)
    
    def tanh_prime(x: Tensor) -> Tensor:
        y = tanh(x)
        return 1 - y ** 2
    

class Tanh(Activation):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)
