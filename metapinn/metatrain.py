# -----------------------------------------------------------------------------
# Author: Shijun Cheng
# Contact Email: sjcheng.academic@gmail.com
# Description: Meta-training script for Meta-PINN. This code performs 
#              meta-training using a MAML-like framework for PDE-based tasks.
#              It loads training data, performs inner and outer optimization 
#              loops, logs progress, periodically saves checkpoints, and 
#              includes validation and testing steps.
# -----------------------------------------------------------------------------

import  os
import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import  argparse
import  scipy.io as scio
from torch.utils.tensorboard import SummaryWriter
from pinnmodel import PINN
from copy import deepcopy
from random import shuffle
import math
from collections import OrderedDict
import random
from losses import PINNLoss, RegLoss
from utils import PositionalEncod, calculate_grad

# Directory for saving checkpoints
dir_checkpoints = './checkpoints/metatrain/'
os.makedirs(dir_checkpoints, exist_ok=True)

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    
    print(args)

    # Set up TensorBoard writer for logging training/validation metrics
    writer = SummaryWriter(log_dir=f'./runs/metatrain/MetaLR_{args.meta_lr}_UpdateLR_{args.update_lr}_Epoch_{args.epoch}_Updatestep_{args.update_step}')

    # Define the training device
    device = torch.device('cuda')

    # pinn model
    input_dim = 3   # Input dimension (e.g., x, z, sx)
    output_dim = 2  # Output dimension (e.g., real and imaginary parts)
    neurons = [input_dim, 256, 256, 128, 128, 64, 64, output_dim]

    # Initialize the meta-training model (MAML-like structure)
    maml = PINN(neurons).to(device)

    # Count the total number of trainable parameters
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    omega = 2*math.pi*args.omega

    # -------------------- Load Training Data --------------------
    # Load support data set for meta-training
    data_spt = scio.loadmat('../dataset/metatrain/train_spt.mat')
    x_spt = data_spt['x_spt']
    sx_spt = data_spt['sx_spt']
    z_spt = data_spt['z_spt']
    m_spt = data_spt['m_spt']
    m0_spt = data_spt['m0_spt']
    U0_real_spt = data_spt['U0_real_spt']
    U0_imag_spt = data_spt['U0_imag_spt']

    # Load query data set for meta-training
    data_qry = scio.loadmat('../dataset/metatrain/train_qry.mat')
    x_qry = data_qry['x_qry']
    sx_qry = data_qry['sx_qry']
    z_qry = data_qry['z_qry']
    m_qry = data_qry['m_qry']
    m0_qry = data_qry['m0_qry']
    U0_real_qry = data_qry['U0_real_qry']
    U0_imag_qry = data_qry['U0_imag_qry']
    
    # Convert numpy arrays to torch tensors and move them to GPU
    x_spt = torch.tensor(x_spt, dtype=torch.float32, requires_grad=True).to(device)
    z_spt = torch.tensor(z_spt, dtype=torch.float32, requires_grad=True).to(device)
    sx_spt = torch.tensor(sx_spt, dtype=torch.float32).to(device)
    m_spt, m0_spt = torch.tensor(m_spt, dtype=torch.float32).to(device), \
                    torch.tensor(m0_spt, dtype=torch.float32).to(device)
    U0_real_spt, U0_imag_spt = torch.tensor(U0_real_spt, dtype=torch.float32).to(device), \
                               torch.tensor(U0_imag_spt, dtype=torch.float32).to(device)

    x_qry = torch.tensor(x_qry, dtype=torch.float32, requires_grad=True).to(device)
    z_qry = torch.tensor(z_qry, dtype=torch.float32, requires_grad=True).to(device)
    sx_qry = torch.tensor(sx_qry, dtype=torch.float32).to(device)
    m_qry, m0_qry = torch.tensor(m_qry, dtype=torch.float32).to(device), \
                    torch.tensor(m0_qry, dtype=torch.float32).to(device)
    U0_real_qry, U0_imag_qry = torch.tensor(U0_real_qry, dtype=torch.float32).to(device), \
                               torch.tensor(U0_imag_qry, dtype=torch.float32).to(device)


    # -------------------- Load Test Data --------------------
    # training data set
    data_test_train = scio.loadmat('../dataset/metatest/train_data.mat')
    x_test_train = data_test_train['x_train']
    sx_test_train = data_test_train['sx_train']
    z_test_train = data_test_train['z_train']
    m_test = data_test_train['m_train']
    m0_test = data_test_train['m0_train']
    U0_real_test = data_test_train['U0_real_train']
    U0_imag_test = data_test_train['U0_imag_train']

    # testing data set
    data_test_test = scio.loadmat('../dataset/metatest/test_data.mat')
    x_test_test = data_test_test['x_star']
    sx_test_test = data_test_test['sx_star']
    z_test_test = data_test_test['z_star']
    dU_real_test = data_test_test['dU_real_star']
    dU_imag_test = data_test_test['dU_imag_star']

    # Convert data to torch tensors
    x_test_train = torch.tensor(x_test_train, dtype=torch.float32, requires_grad=True).to(device)
    z_test_train = torch.tensor(z_test_train, dtype=torch.float32, requires_grad=True).to(device)
    sx_test_train = torch.tensor(sx_test_train, dtype=torch.float32).to(device)
    m_test, m0_test = torch.tensor(m_test, dtype=torch.float32).to(device), \
                      torch.tensor(m0_test, dtype=torch.float32).to(device)
    U0_real_test, U0_imag_test = torch.tensor(U0_real_test, dtype=torch.float32).to(device), \
                                 torch.tensor(U0_imag_test, dtype=torch.float32).to(device)

    x_test_test = torch.tensor(x_test_test, dtype=torch.float32, requires_grad=True).to(device)
    z_test_test = torch.tensor(z_test_test, dtype=torch.float32, requires_grad=True).to(device)
    sx_test_test = torch.tensor(sx_test_test, dtype=torch.float32).to(device)
    dU_real_test, dU_imag_test = torch.tensor(dU_real_test, dtype=torch.float32).to(device), \
                                 torch.tensor(dU_imag_test, dtype=torch.float32).to(device)

    data_len, task_num = x_spt.size()

    # Set up the optimizer for meta-training
    meta_optimizer = torch.optim.AdamW(maml.parameters(), lr=args.meta_lr, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, step_size=5000, gamma=0.8)

    # Define loss functions
    criterion_pde = PINNLoss()           # PDE residual loss (physics-informed)
    criterion_reg = RegLoss()            # Additional regularization loss

    # Set up the optimizer for validation
    maml_copy = deepcopy(maml)
    test_optimizer = torch.optim.AdamW(maml_copy.parameters(), lr=args.test_lr, weight_decay=4e-5)

    # -------------------- Meta-Training Loop --------------------
    for step in range(args.epoch):
        maml.train()

        rand_task_spt = list(range(0,task_num))
        rand_task_qry = list(range(0,task_num))
        shuffle(rand_task_spt)
        shuffle(rand_task_qry)

        outer_loss = torch.tensor(0., device=device)

        # Loop over tasks to compute outer_loss for meta-update
        for i in range(task_num):

            # Randomly sample support (inner update) dataset
            x = x_spt[:, rand_task_spt[i]].unsqueeze(1)
            z = z_spt[:, rand_task_spt[i]].unsqueeze(1)
            sx = sx_spt[:, rand_task_spt[i]].unsqueeze(1)

            m0 = m0_spt[:, rand_task_spt[i]].unsqueeze(1)
            m = m_spt[:, rand_task_spt[i]].unsqueeze(1)
            U0_real = U0_real_spt[:, rand_task_spt[i]].unsqueeze(1)
            U0_imag = U0_imag_spt[:, rand_task_spt[i]].unsqueeze(1)

            # Get a copy of the current model parameters for inner updates
            params = OrderedDict(maml.named_parameters())

            # Loop over tasks to compute outer_loss for meta-update
            for k in range(args.update_step):
                input = torch.cat([x,z,sx],-1)

                # Forward with functional parameters
                du_real, du_imag = maml.functional_forward(input, params=params)

                # Compute gradients (second derivatives) for PDE constraints
                du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x, z, du_real, du_imag)

                # Compute PDE loss and regularization losses
                loss_pde = criterion_pde(x, z, sx, omega, m, m0, U0_real, U0_imag, du_real, du_imag, du_real_xx, 
                    du_real_zz, du_imag_xx, du_imag_zz)

                loss_reg = criterion_reg(x, z, sx, omega, m0, du_real, du_imag)

                inner_loss = args.loss_scale * (loss_pde + loss_reg)

                # Compute gradients of inner_loss w.r.t. params
                grads = torch.autograd.grad(inner_loss, params.values(), create_graph=not args.first_order)

                # Update params for this task's inner loop (fast adaptation)
                params = OrderedDict(
                        (name, param - args.update_lr * grad)
                        for ((name, param), grad) in zip(params.items(), grads))

            # -------------------- Outer Update Step --------------------
            # Randomly sample query (outer update) dataset
            x = x_qry[:, rand_task_qry[i]].unsqueeze(1)
            z = z_qry[:, rand_task_qry[i]].unsqueeze(1)
            sx = sx_qry[:, rand_task_qry[i]].unsqueeze(1)

            m0 = m0_qry[:, rand_task_qry[i]].unsqueeze(1)
            m = m_qry[:, rand_task_qry[i]].unsqueeze(1)
            U0_real = U0_real_qry[:, rand_task_qry[i]].unsqueeze(1)
            U0_imag = U0_imag_qry[:, rand_task_qry[i]].unsqueeze(1)

            input = torch.cat([x,z,sx],-1)

            # Evaluate on the query set with updated parameters
            du_real, du_imag = maml.functional_forward(input, params=params)

            du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x, z, du_real, du_imag)

            # Compute losses on the query set
            loss_pde = criterion_pde(x, z, sx, omega, m, m0, U0_real, U0_imag, du_real, du_imag, du_real_xx, 
                    du_real_zz, du_imag_xx, du_imag_zz)

            loss_reg = criterion_reg(x, z, sx, omega, m0, du_real, du_imag)

            # Accumulate outer loss from all tasks
            outer_loss += args.loss_scale * (loss_pde + loss_reg)

        # Average outer_loss over the number of tasks
        outer_loss = outer_loss / task_num

        # Meta-optimizer update on outer_loss
        meta_optimizer.zero_grad()
        outer_loss.backward()
        meta_optimizer.step()
        scheduler.step()

        # Logging to TensorBoard
        writer.add_scalar('Loss/meta_loss', outer_loss.item(), step)
        writer.add_scalar('Loss/loss_pde', loss_pde.item(), step)
        writer.add_scalar('Loss/loss_reg', loss_reg.item(), step)
        writer.add_scalar('Loss/inner_loss', inner_loss.item(), step)

        # Print training status every 100 steps
        if (step + 1) % 100 == 0:
            print(f'step: {step + 1} Training inner loss: {inner_loss.item()}')
            print(f'step: {step + 1} Training meta loss: {outer_loss.item()}')
            print(f'step: {step + 1} Training PDE loss: {loss_pde.item()}')
            print(f'step: {step + 1} Training REG loss: {loss_reg.item()}')

        # Save model checkpoint every 100 steps
        if (step + 1) % 100 == 0:
            torch.save(maml.state_dict(), f'{dir_checkpoints}CP_epoch{step + 1}.pth')

        # Perform validation every 5000 steps
        if (step + 1) % 5000 == 0:
            print('---------------------------------------------------------')
            print('------------------- Validation start --------------------')
            print('---------------------------------------------------------')

            # copy a meta-trained model
            maml_copy.load_state_dict(maml.state_dict())
            maml_copy.train()

            # Fine-tune on validation data
            for k in range(args.update_step_test):
                test_optimizer.zero_grad()

                input = torch.cat([x_test_train,z_test_train,sx_test_train],-1)

                du_real, du_imag = maml_copy(input)

                du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x_test_train, z_test_train, du_real, du_imag)

                loss_pde = criterion_pde(x_test_train, z_test_train, sx_test_train, omega, m_test, m0_test,  
                    U0_real_test, U0_imag_test, du_real, du_imag, du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)

                loss_reg = criterion_reg(x_test_train, z_test_train, sx_test_train, omega, m_test, m0_test,  
                    U0_real_test, U0_imag_test, du_real, du_imag, du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)

                test_loss = loss_pde + loss_reg

                test_loss.backward()

                test_optimizer.step()

                if (k + 1) % 100 == 0:
                    print(f'step: {k + 1} Validation loss: {test_loss.item()}')
                    print(f'step: {k + 1} Validation PDE loss: {loss_pde.item()}')
                    print(f'step: {k + 1} Validation REG loss: {loss_reg.item()}')

            # Evaluate on test data after validation
            with torch.no_grad():
                input = torch.cat([x_test_test,z_test_test,sx_test_test],-1)
                du_real, du_imag = maml_copy(input)
                accs_real = (torch.pow((dU_real_test-du_real),2)).mean().item()
                accs_imag = (torch.pow((dU_imag_test-du_imag),2)).mean().item()

            print(f'Test accs real: {accs_real}')
            print(f'Test accs imag: {accs_imag}')

            print('---------------------------------------------------------')
            print('---------------------- Ending Test ----------------------')
            print('---------------------------------------------------------')

    writer.close()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50000)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=2e-3)
    argparser.add_argument('--test_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--omega', type=float, help='frequecy', default=5)
    argparser.add_argument('--loss_scale', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=20)
    argparser.add_argument('--update_step_test', type=int, help='test upate step', default=5000)
    argparser.add_argument('--PosEnc', type=int, help='PosEnc', default=2)
    argparser.add_argument('--first_order', type=str, help='whether first order approximation of MAML is used', default=True)

    args = argparser.parse_args()

    main(args)
