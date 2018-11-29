import h5py
import os.path
import torch
import numpy as np

def save_params(destination, path, params):
    if path not in destination:
        destination.create_dataset(path, data=params, compression='gzip')
    else:
        destination[path][...] = params
    return

def save_model(model, destination):
    if 'model' not in destination:
        destination.create_group('model')
    for name, value in model.state_dict().items():
        save_params(destination, 'model/'+name, value)
    return

def save_epoch(epoch, destination):
    if 'training' not in destination:
        destination.create_dataset('training/epoch', data=epoch)
    else:
        destination['training/epoch'][()] = epoch
    return

def checkpoint(model, epoch, optimizer, dest, exp_folder):
    save_model(model, dest)
    #save_epoch(epoch, dest)
    torch.save(optimizer.state_dict(),
               os.path.join(exp_folder, 'checkpoint.opt'))
    return
