import os.path
import argparse
import math
import torch
import torchvision.utils as vutils
from datetime import datetime
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from shapes_3d import Shapes3D
from slate import SLATE

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--image_size', type=int, default=64)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--log_path', default='logs')
parser.add_argument('--data_path', default='3dshapes.h5')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_main', type=float, default=1e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)

parser.add_argument('--num_dec_blocks', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=1024)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=3)
parser.add_argument('--num_slot_heads', type=int, default=1)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--pos_channels', type=int, default=4)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--hard', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

train_dataset = Shapes3D(root=args.data_path, phase='train')
val_dataset = Shapes3D(root=args.data_path, phase='val')

train_sampler = None
val_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5

model = SLATE(args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    stagnation_counter = checkpoint['stagnation_counter']
    lr_decay_factor = checkpoint['lr_decay_factor']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0
    stagnation_counter = 0
    lr_decay_factor = 1.0

model = model.cuda()

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'dvae' not in x[0]), 'lr': args.lr_main},
])
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])


def linear_warmup(step, start_value, final_value, start_step, final_step):
    
    assert start_value <= final_value
    assert start_step <= final_step
    
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b
    
    return value


def cosine_anneal(step, start_value, final_value, start_step, final_step):
    
    assert start_value >= final_value
    assert start_step <= final_step
    
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b
    
    return value


def visualize(image, recon_orig, gen, attns, N=8):
    _, _, H, W = image.shape
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    recon_orig = recon_orig[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    gen = gen[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    attns = attns[:N].expand(-1, -1, 3, H, W)

    return torch.cat((image, recon_orig, gen, attns), dim=1).view(-1, 3, H, W)


for epoch in range(start_epoch, args.epochs):
    
    model.train()
    
    for batch, image in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch
        
        tau = cosine_anneal(
            global_step,
            args.tau_start,
            args.tau_final,
            0,
            args.tau_steps)

        lr_warmup_factor = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps)

        optimizer.param_groups[0]['lr'] = lr_decay_factor * args.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor * args.lr_main

        image = image.cuda()

        optimizer.zero_grad()

        (recon, cross_entropy, mse, attns) = model(image, tau, args.hard)
        
        loss = mse + cross_entropy
        
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer.step()

        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, batch, train_epoch_size, loss.item(), mse.item()))
                
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)
                
                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_main', optimizer.param_groups[1]['lr'], global_step)

    with torch.no_grad():
        gen_img = model.reconstruct_autoregressive(image[:32])
        vis_recon = visualize(image, recon, gen_img, attns, N=32)
        grid = vutils.make_grid(vis_recon, nrow=args.num_slots + 3, pad_value=0.2)[:, 2:-2, 2:-2]
        writer.add_image('TRAIN_recon/epoch={:03}'.format(epoch+1), grid)
    
    with torch.no_grad():
        model.eval()
        
        val_cross_entropy_relax = 0.
        val_mse_relax = 0.
        
        val_cross_entropy = 0.
        val_mse = 0.
        
        for batch, image in enumerate(val_loader):
            image = image.cuda()

            (recon_relax, cross_entropy_relax, mse_relax, attns_relax) = model(image, tau, False)
            
            (recon, cross_entropy, mse, attns) = model(image, tau, True)
            
            val_cross_entropy_relax += cross_entropy_relax.item()
            val_mse_relax += mse_relax.item()
            
            val_cross_entropy += cross_entropy.item()
            val_mse += mse.item()

        val_cross_entropy_relax /= (val_epoch_size)
        val_mse_relax /= (val_epoch_size)
        
        val_cross_entropy /= (val_epoch_size)
        val_mse /= (val_epoch_size)
        
        val_loss_relax = val_mse_relax + val_cross_entropy_relax
        val_loss = val_mse + val_cross_entropy

        writer.add_scalar('VAL/loss_relax', val_loss_relax, epoch+1)
        writer.add_scalar('VAL/cross_entropy_relax', val_cross_entropy_relax, epoch + 1)
        writer.add_scalar('VAL/mse_relax', val_mse_relax, epoch+1)

        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        writer.add_scalar('VAL/cross_entropy', val_cross_entropy, epoch + 1)
        writer.add_scalar('VAL/mse', val_mse, epoch+1)

        print('====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F}'.format(
            epoch+1, val_loss, val_mse))

        if val_loss < best_val_loss:
            stagnation_counter = 0
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

            if 50 <= epoch:
                gen_img = model.reconstruct_autoregressive(image)
                vis_recon = visualize(image, recon, gen_img, attns, N=32)
                grid = vutils.make_grid(vis_recon, nrow=args.num_slots + 3, pad_value=0.2)[:, 2:-2, 2:-2]
                writer.add_image('VAL_recon/epoch={:03}'.format(epoch + 1), grid)

        else:
            stagnation_counter += 1
            if stagnation_counter >= args.patience:
                lr_decay_factor = lr_decay_factor / 2.0
                stagnation_counter = 0

        writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stagnation_counter': stagnation_counter,
            'lr_decay_factor': lr_decay_factor,
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()
