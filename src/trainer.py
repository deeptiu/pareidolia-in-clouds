from __future__ import print_function

import torch
import torch.nn.functional as F
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


def train(model, optimizer, criterion, scheduler=None, model_name='model'):
    
    epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)


    train_dataset = CloudDataset("images/", size=512)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

    cnt = 0
    for epoch in range(epochs):
        total_images = 0
        total_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            total_images += len(data)

            optimizer.zero_grad()
    
            # Forward pass
            output = model(data)
            output_labels = torch.argmax(output, dim=1)

            total_correct += torch.eq(output_labels, target).sum()

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            print(f"EPOCH: {epoch}, BATCH: {batch_idx}, LOSS: {loss.item()}")

            # if scheduler is not None:
            #     scheduler.step()
            #     lr = optimizer.param_groups[0]['lr']
            #     # writer.add_scalar('Learning Rate', lr, cnt)
            
        train_accuracy = total_correct / total_images
        print(f"EPOCH: {epoch}, ACCURACY: {train_accuracy}")

        # save model
        save_model(epoch, model_name, optimizer, model)