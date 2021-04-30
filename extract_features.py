import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow', default='flow')
parser.add_argument('-load_model', type=str, default='./models/flow_charades.pt')
parser.add_argument('-root', type=str, default='./flows/train')
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-save_dir', type=str, default='flow_feature')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset


def run(max_steps=64e3, mode='flow', root='./frames', split='gt.json', batch_size=1,
        load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                             pin_memory=True)

    val_dataset = Dataset(split, 'test', root, mode, test_transforms, num=-1, save_dir=save_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(20, in_channels=2)
    else:
        i3d = InceptionI3d(20, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    for phase in ['train', 'val']:
        i3d.train(False)  # Set model to evaluate mode

        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0

        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels, name = data
            # if os.path.exists(os.path.join(save_dir, name[0] + '.npy')):
            #     continue

            b, c, t, h, w = inputs.shape
            if t > 16:
                features = []
                for start in range(0, t, 16):
                    end = min(t - 1, start + 16)
                    if end < start + 16:
                        break
                    # start = max(1, start - 48)
                    ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda(), volatile=True)
                    feature = i3d.extract_features(ip)
                    feature = torch.squeeze(feature)
                    features.append(feature.data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.asarray(features))
            else:
                # wrap them in Variable
                inputs = Variable(inputs.cuda(), volatile=True)
                features = i3d.extract_features(inputs)
                np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
