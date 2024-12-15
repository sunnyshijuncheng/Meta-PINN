import torch
from torch import nn
import numpy as np

class PositionalEncod(nn.Module):
    def __init__(self, PosEnc=2, device='cpu'):
        """
        Positional Encoding module:
        Applies sinusoidal/cosine positional encodings to the input coordinates (x, z, sx).
        
        Arguments:
        - PosEnc: Defines how many frequency bands (powers of 2) to use in the encoding.
        - device: The device on which to create the encoding tensors.
        
        This creates three sets of frequency multipliers (for x, z, sx) and applies 
        sin and cos transformations to enhance feature representation.
        """
        super().__init__()
        self.PEnc = PosEnc
        # Create frequency multiplier arrays for x, z, sx
        self.k_pi_x = (torch.tensor(np.pi)*(2**torch.arange(self.PEnc))).reshape(-1, self.PEnc).to(device); self.k_pi_x = self.k_pi_x.T
        self.k_pi_z = (torch.tensor(np.pi)*(2**torch.arange(self.PEnc))).reshape(-1, self.PEnc).to(device); self.k_pi_z = self.k_pi_z.T
        self.k_pi_sx = (torch.tensor(np.pi)*(2**torch.arange(self.PEnc))).reshape(-1, self.PEnc).to(device); self.k_pi_sx = self.k_pi_sx.T

    def forward(self, input):
        """
        input: Tensor of shape (N, 3), where columns are x, z, sx.
        This method applies sin and cos transformations for each coordinate with different frequencies.
        
        Returns:
        - Tensor with original input features concatenated with positional encodings.
          If input is (N, 3), output will have more dimensions due to encoded features.
        """
        # Apply sin and cos encoding for x
        tmpx = torch.cat([torch.sin(self.k_pi_x*input[:,0]), torch.cos(self.k_pi_x*input[:,0])], axis=0)
        # Apply sin and cos encoding for z
        tmpz = torch.cat([torch.sin(self.k_pi_z*input[:,1]), torch.cos(self.k_pi_z*input[:,1])], axis=0)
        # Apply sin and cos encoding for sx
        tmpsx = torch.cat([torch.sin(self.k_pi_sx*input[:,2]), torch.cos(self.k_pi_sx*input[:,2])], axis=0)

        # Concatenate all encodings: shape (2*PosEnc*3, N), then transpose to (N, ...)
        cat = torch.cat((tmpx, tmpz, tmpsx), axis=0)
        return torch.cat([input, cat.T], -1)

def normalizer(x, dmin, dmax):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def calculate_grad(x, z, du_real, du_imag):
    """
    Calculates second-order derivatives of the predicted real and imaginary wavefields w.r.t. x and z.
    
    Inputs:
    - x, z: spatial coordinates (tensors that require grad)
    - du_real, du_imag: predicted real and imaginary parts of the wavefield
    
    This function computes:
    du_real_xx = d²u_real/dx²
    du_real_zz = d²u_real/dz²
    du_imag_xx = d²u_imag/dx²
    du_imag_zz = d²u_imag/dz²

    Returns:
    - du_real_xx, du_real_zz, du_imag_xx, du_imag_zz: second-order derivatives w.r.t. x and z.
    """
    # First derivatives for real field
    du_real_x = torch.autograd.grad(du_real, x, grad_outputs=torch.ones_like(du_real), create_graph=True)[0]
    du_real_z = torch.autograd.grad(du_real, z, grad_outputs=torch.ones_like(du_real), create_graph=True)[0]
    # Second derivatives for real field
    du_real_xx = torch.autograd.grad(du_real_x, x, grad_outputs=torch.ones_like(du_real_x), create_graph=True)[0]
    du_real_zz = torch.autograd.grad(du_real_z, z, grad_outputs=torch.ones_like(du_real_z), create_graph=True)[0]

    # First derivatives for imaginary field
    du_imag_x = torch.autograd.grad(du_imag, x, grad_outputs=torch.ones_like(du_imag), create_graph=True)[0]
    du_imag_z = torch.autograd.grad(du_imag, z, grad_outputs=torch.ones_like(du_imag), create_graph=True)[0]
    # Second derivatives for imaginary field
    du_imag_xx = torch.autograd.grad(du_imag_x, x, grad_outputs=torch.ones_like(du_imag_x), create_graph=True)[0]
    du_imag_zz = torch.autograd.grad(du_imag_z, z, grad_outputs=torch.ones_like(du_imag_z), create_graph=True)[0]

    return du_real_xx, du_real_zz, du_imag_xx, du_imag_zz