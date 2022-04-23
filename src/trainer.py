from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils
from cloud_dataset import CloudDataset


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, optimizer, model):
    # TODO: Q2.2 Implement code for model saving
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "saved_models/"+filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    # TODO Q1.5: Initialize your tensorboard writer here!
    writer = SummaryWriter()
    train_loader = utils.get_data_loader(
        'cloud', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'cloud', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
#             import pdb; pdb.set_trace()
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)
#             print(wgt.shape)
            optimizer.zero_grad()
    
            # Forward pass
            output = model(data)
            # Calculate the loss
            # TODO Q1.4: your loss for multi-label classification
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target, weight=wgt).to(args.device)

            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # TODO Q1.5: Log training loss to tensorboard
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                # writer.add_scalar('Loss', loss.item(), cnt)

            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, mAP = utils.eval_dataset_map(model, args.device, test_loader)
                print(mAP)
                # TODO Q1.5: Log MAP to tensorboard
                # writer.add_scalar('Mean Average Precision', mAP, cnt)
                
                model.train()
            cnt += 1

        # TODO Q3.2: Log Learning rate
        if scheduler is not None:
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            # writer.add_scalar('Learning Rate', lr, cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, optimizer, model)
            
    writer.close()
    
    # Validation iteration
    test_loader = utils.get_data_loader('cloud', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map