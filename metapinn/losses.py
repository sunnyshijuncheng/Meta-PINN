import torch
from torch import nn
from torch.nn import functional as F

class PINNLoss(nn.Module):
    def __init__(self):
        super(PINNLoss,self).__init__()

    def forward(self, x, z, sx, omega, m, m0, u0_real, u0_imag, du_real, du_imag, 
            du_real_xx, du_real_zz, du_imag_xx, du_imag_zz):
        """
        Physics-Informed Neural Network (PINN) loss for a certain PDE:
        This computes a PDE residual loss based on the given second derivatives and wave equation parameters.

        Inputs:
        - x, z, sx: Spatial coordinates and possibly a source location (sx)
        - omega: Angular frequency of the wave
        - m, m0: Model parameters (e.g., wave velocity models, background model m0)
        - u0_real, u0_imag: Reference (background) wavefield solutions (real and imaginary parts)
        - du_real, du_imag: Predicted wavefield perturbations (real and imaginary parts)
        - du_real_xx, du_real_zz: Second derivatives of du_real w.r.t. x and z
        - du_imag_xx, du_imag_zz: Second derivatives of du_imag w.r.t. x and z

        The PDE is typically of the form:
            (omega^2 * m) * du + Δu + omega^2*(m - m0)*u0 = 0
        where Δ is the Laplacian operator (u_xx + u_zz).

        The loss is computed as the L2 norm of the PDE residual for both real and imaginary parts.

        Returns:
        - loss_pde: A scalar representing the PDE residual error.
        """
        loss_real = omega*omega*m*du_real + du_real_xx + du_real_zz + omega*omega*(m-m0)*u0_real
        loss_imag = omega*omega*m*du_imag + du_imag_xx + du_imag_zz + omega*omega*(m-m0)*u0_imag
        
        # Compute combined loss for both real and imaginary parts
        loss_pde = torch.sqrt((torch.pow(loss_real,2)).mean() + (torch.pow(loss_imag,2)).mean())
        return loss_pde

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss,self).__init__()

    def forward(self, x, z, sx, omega, m0, du_real, du_imag):
        """
        Regularization loss:
        This loss impose additional constraints or penalties on the predicted wavefields,
        especially in regions defined by some factor (factor_d) related to the wave equation parameters.

        Inputs:
        - x, z, sx: Spatial coordinates and source position
        - omega: Angular frequency
        - m0: Background model parameter
        - du_real, du_imag: Predicted wavefield perturbations

        The factor_d term seems to be a function that penalizes regions inside a certain radius 
        (e.g., within a certain circle defined by (sx - x)^2 + (z - 0.025)^2) and scales with frequency and velocity.

        The regularization encourages the predicted fields (du_real, du_imag) to remain small or well-behaved in that region.

        Returns:
        - loss_reg: A scalar representing the regularization penalty.
        """
        # factor_d defines a region in space where we impose stronger regularization
        factor_d = F.relu((torch.sqrt(1/m0)*0.5*3.14/omega)**2-(sx-x)**2-(z-0.025)**2)*10e7*omega

        # Compute the regularization as L2 norm within the defined region
        loss_reg = torch.sqrt((factor_d*torch.pow(du_real,2)).mean() + (factor_d*torch.pow(du_imag,2)).mean())
        return loss_reg
