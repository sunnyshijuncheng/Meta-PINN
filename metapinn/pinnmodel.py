"""
This script defines a PINN designed to represent the scattered wavefield. 
The network is built using PyTorch and consists of fully-connected layers with sine activation functions.

Key Components:
- Basicblock: A fundamental building block consisting of a linear layer followed by a sine activation.
- PINN: The main neural network class that assembles multiple Basicblocks and includes both standard
        and functional forward passes. The functional forward pass is particularly useful for
        meta-learning scenarios such as Model-Agnostic Meta-Learning (MAML), allowing the network
        to adapt quickly to new tasks with externally provided parameters.
- Functional_linear: A helper function that applies a linear transformation followed by a sine activation, used within the functional forward pass.

Usage:
- Define the architecture by specifying the number of neurons in each layer.
- Instantiate the PINN model with the desired layer configuration.
- Use the `forward` method for standard predictions or `functional_forward` for meta-learning tasks.

Example:
    layers = [input_dim, hidden1_dim, hidden2_dim, output_dim]
    model = PINN(layers)
    output1, output2 = model(input_tensor)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Basic fully-connected layer with sine activation
class Basicblock(nn.Module):
    def __init__(self, in_planes, out_planes):
        """
        Initializes the Basicblock module.

        Arguments:
        - in_planes: Number of input features.
        - out_planes: Number of output features.
        """
        super(Basicblock, self).__init__()
        # Define a linear layer that maps input features to output features
        self.layer = nn.Linear(in_planes, out_planes)

    def forward(self, x):
        """
        Defines the forward pass of the Basicblock.

        Arguments:
        - x: Input tensor.

        Returns:
        - Output tensor after applying linear transformation and sine activation.
        """
        # Apply the linear layer
        out = self.layer(x)
        # Apply the sine activation function
        out = torch.sin(out)
        return out

# Vanilla Physics-Informed Neural Network (PINN)
class PINN(nn.Module):
    def __init__(self, layers):
        """
        Initializes a vanilla PINN model.

        Arguments:
        - layers: A list specifying the number of neurons in each layer.
                  Example: [input_dim, hidden1_dim, ..., output_dim]
        """
        super(PINN, self).__init__()
        self.layers = layers
        self.in_planes = self.layers[0]
        # Create hidden layers using Basicblock, excluding the last layer
        self.layer1 = self._make_layer(Basicblock, self.layers[1:len(layers)-1])
        # Define the final linear layer mapping to the output dimension
        self.layer2 = nn.Linear(layers[-2], layers[-1])

    def _make_layer(self, block, layers):
        """
        Constructs multiple Basicblock layers.

        Arguments:
        - block: The block class to use (e.g., Basicblock).
        - layers: A list of output dimensions for each block.

        Returns:
        - A sequential container of the specified blocks.
        """
        layers_net = []
        for layer in layers:
            # Add a Basicblock with current input and specified output dimensions
            layers_net.append(block(self.in_planes, layer))
            # Update in_planes for the next layer
            self.in_planes = layer
        return nn.Sequential(*layers_net)

    def forward(self, x):
        """
        Defines the main forward pass of the PINN.

        Arguments:
        - x: Input tensor, e.g., containing features like x, z, sx.

        Returns:
        - Two output channels: x[:, 0:1] and x[:, 1:2].
        """
        # Pass through the hidden layers
        out = self.layer1(x)
        # Pass through the final linear layer
        out = self.layer2(out)
        # Return the two specified output channels
        return out[:, 0:1], out[:, 1:2]

    def functional_forward(self, x, params):
        """
        Performs a functional forward pass using externally provided parameters.
        Useful for meta-learning approaches like MAML.

        Arguments:
        - x: Input features.
        - params: A dictionary containing weights and biases to use instead of the model's internal parameters.

        Returns:
        - Two output channels: x[:, 0:1] and x[:, 1:2].
        """
        # Iterate through all hidden layers except the last one
        for id in range(len(self.layers) - 2):
            # Retrieve the weight and bias for the current layer from params
            weight = params[f'layer1.{id}.layer.weight']
            bias = params[f'layer1.{id}.layer.bias']
            # Apply the functional linear layer with sine activation
            x = Functional_linear(x, weight, bias)
        # Apply the final linear layer using params
        x = F.linear(x, params['layer2.weight'], params['layer2.bias'])
        # Return the two specified output channels
        return x[:, 0:1], x[:, 1:2]

def Functional_linear(inp, weight, bias):
    """
    Applies a functional linear transformation followed by a sine activation.
    Equivalent to: F.linear(inp, weight, bias) -> sin.

    Arguments:
    - inp: Input tensor.
    - weight: Weight tensor for the linear layer.
    - bias: Bias tensor for the linear layer.

    Returns:
    - Output tensor after linear transformation and sine activation.
    """
    # Apply the linear transformation
    inp = F.linear(inp, weight, bias)
    # Apply the sine activation function
    inp = torch.sin(inp)
    return inp
