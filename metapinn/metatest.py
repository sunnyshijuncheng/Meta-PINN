# -----------------------------------------------------------------------------
# Author: Shijun Cheng
# Description:
# This script performs meta-testing (or validation/testing phase) for the 
# meta-trained Meta-PINN model. It:
# 1. Loads a pre-trained model checkpoint (from meta-training).
# 2. Loads validation and test data.
# 4. Fine-tunes the model on the validation set.
# 5. Periodically evaluates the model on the test set and saves predictions, logs, and checkpoints.
# -----------------------------------------------------------------------------

import  os
import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import  argparse
from torch.autograd import Variable
from pinnmodel import PINN
import  scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import math
from losses import PINNLoss, RegLoss
from utils import PositionalEncod, calculate_grad

# Paths to meta-trained checkpoint, meta-testing checkpoint, output directories, etc.
dir_meta = './checkpoints/metatrain/meta_trained.pth'
dir_checkpoints = './checkpoints/metatest/'
dir_output = './output/'
os.makedirs(dir_output, exist_ok=True)
os.makedirs(dir_checkpoints, exist_ok=True)

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    
    print(args)

    # Define the training device
    device = torch.device('cuda')

    # Set up TensorBoard writer for logging fine-tuning and testing results
    writer = SummaryWriter(log_dir=f'runs/metatest/LR_{args.lr}_Epoch_{args.epoch}')

    # Define input/output dimensions for the model
    input_dim = 3     # (x, z, sx)
    output_dim = 2    # (real part, imaginary part of the wavefield)

    # A list specifying the number of neurons in each layer.
    neurons = [input_dim, 256, 256, 128, 128, 64, 64, output_dim]
    Net = PINN(neurons).to(device)

    if args.use_meta:
        Net.load_state_dict(torch.load(dir_meta, map_location=device))
        print(f'load model from meta_trained model {dir_meta}')
    else:
        print(f'model start from random initialization')

    tmp = filter(lambda x: x.requires_grad, Net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    omega = 2*math.pi*args.omega

    # Load validation (train_data) from .mat file
    data = sio.loadmat('../dataset/metatest/train_data.mat')
    x = data['x_train']
    z = data['z_train']
    sx = data['sx_train']
    m = data['m_train']
    m0 = data['m0_train']
    U0_real = data['U0_real_train']
    U0_imag = data['U0_imag_train']

    # train data set
    data = sio.loadmat('../dataset/metatest/test_data.mat')
    x_test = data['x_star']
    sx_test = data['sx_star']
    z_test = data['z_star']
    dU_real = data['dU_real_star']
    dU_imag = data['dU_imag_star']

    # Convert numpy arrays to torch tensors and move them to the GPU
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    z = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(device)
    sx = torch.tensor(sx, dtype=torch.float32).to(device)
    m, m0 = torch.tensor(m, dtype=torch.float32).to(device), \
            torch.tensor(m0, dtype=torch.float32).to(device)
    U0_real, U0_imag = torch.tensor(U0_real, dtype=torch.float32).to(device), \
                       torch.tensor(U0_imag, dtype=torch.float32).to(device)

    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    z_test = torch.tensor(z_test, dtype=torch.float32).to(device)
    sx_test = torch.tensor(sx_test, dtype=torch.float32).to(device)
    dU_real, dU_imag = torch.tensor(dU_real, dtype=torch.float32).to(device), \
                       torch.tensor(dU_imag, dtype=torch.float32).to(device)

    input_test = torch.cat([x_test,z_test,sx_test],-1)

    # Set up optimizer and learning rate scheduler for validation fine-tuning
    optim = torch.optim.AdamW(Net.parameters(), weight_decay = 4e-5, lr=args.lr)
    milestones = [2000]
    gamma = 0.5
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=gamma)
    print('milestones:',milestones)
    print('gamma:',gamma)

    # Define loss functions
    criterion_pde = PINNLoss()
    criterion_reg = RegLoss()

    for step in range(args.epoch):
        Net.train()
        # optimize theta parameters
        optim.zero_grad()

        input = torch.cat([x,z,sx],-1)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        du_real, du_imag = Net(input)

        du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x, z, du_real, du_imag)

        loss_pde = criterion_pde(x, z, sx, omega, m, m0,  
                U0_real, U0_imag, du_real, du_imag, du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)

        loss_reg = criterion_reg(x, z, sx, omega, m0, du_real, du_imag)

        loss = args.loss_scale*(loss_pde + loss_reg)

        loss.backward()

        optim.step()

        scheduler.step()

        # Evaluate on test data
        with torch.no_grad():
            Net.eval()
            du_real, du_imag = Net(input_test)
            accs_real = (torch.pow((dU_real - du_real),2)).mean().item()
            accs_imag = (torch.pow((dU_imag - du_imag),2)).mean().item()

        writer.add_scalar('Loss/train', loss.item(), step)
        writer.add_scalar('Loss_pde/train', loss_pde.item(), step)
        writer.add_scalar('Loss_reg/train', loss_reg.item(), step)
        writer.add_scalar('accs_real', accs_real, step)
        writer.add_scalar('accs_imag', accs_imag, step)

        if (step + 1) % 100 == 0:
            print(f'step: {step + 1} Training loss: {loss.item()}')
            print(f'step: {step + 1} Training PDE loss: {loss_pde.item()}')
            print(f'step: {step + 1} Training REG loss: {loss_reg.item()}')
            print(f'step: {step + 1} Test accs real: {accs_real}')
            print(f'step: {step + 1} Test accs imag: {accs_imag}')

        if (step + 1) % 100 == 0:
            # Save predictions and accuracy metrics to .mat file
            sio.savemat(f'{dir_output}/pred{step+1}.mat', 
                        {'du_real_pred': du_real.cpu().numpy(), 
                         'accs_real': accs_real, 
                         'du_imag_pred': du_imag.cpu().numpy(), 
                         'accs_imag': accs_imag})

            torch.save(Net.state_dict(), f'{dir_checkpoints}CP_epoch{step + 1}.pth')

    writer.close()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--lr', type=float, help='meta-level outer learning rate', default=1.2e-3)
    argparser.add_argument('--omega', type=float, help='frequecy', default=5)
    argparser.add_argument('--loss_scale', type=float, help='Scaling factor for total losses', default=0.1)
    argparser.add_argument('--PosEnc', type=int, help='PosEnc', default=2)
    argparser.add_argument('--use_meta', type=str, help='whether use meta-trained model', default=True)

    args = argparser.parse_args()

    main(args)
