import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
from torch.utils.tensorboard import SummaryWriter



writer = SummaryWriter('rgb-experiments/log')

# log embeddings
encoding_files = glob.glob('/n/fs/3d-indoor/sdfstudio_outputs/3d-indoor/multiscene/rgb-experiments/downsampled_encodings/*.npy')

for i, fname in enumerate(encoding_files):
    tag = fname.split('/')[-1][:-4]
    features = np.load(fname)

    writer.add_embedding(features,
                         tag=tag,
                         global_step=i,
                         )
writer.close()
