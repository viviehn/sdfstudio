import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import pandas as pd
import sys
import argparse

#writer = SummaryWriter('view/features')

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str)
parser.add_argument('--input', type=str, action='append')
parser.add_argument('--num_samples', type=int)

args = parser.parse_args()

num_samples=args.num_samples


all_features = []
all_input_names = []
for fname in args.input:
    features = np.load(fname)
    print(f'Sampling {len(features)} features down to {num_samples} samples')
    indices = np.random.choice(len(features), num_samples, replace=False)

    features_down = features[indices]
    all_features.append(features_down)
    input_names = [fname for _ in features_down]
    all_input_names.append(input_names)

'''
scene_ids_down = np.array(scene_ids)[indices]
num_train_imgs_down = np.array(num_train_imgs)[indices]
rand_ids_down = np.array(rand_ids)[indices]
#scene_datasets_down= np.array(scene_datasets)[indices]

data_dict = {"feats": features_down.tolist(),
             "scenes": scene_ids_down.tolist(),
             "num_ims": num_train_imgs_down.tolist(),
             #"scene_dataset": scene_datasets_down.tolist(),
             "ids": rand_ids_down.tolist()}
'''

all_features = np.concatenate(all_features)
all_input_names = np.concatenate(all_input_names).flatten()
print(all_features.shape, all_input_names.shape)
data_dict = {"feats": all_features.tolist(),
             "input_file": all_input_names.tolist(),
             }


df = pd.DataFrame(data_dict)

for p in [30.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
    embed = TSNE(n_components=2, learning_rate='auto', verbose=3,
                 perplexity=p)

    embedded_feats = embed.fit_transform(all_features)


    scatter = sns.scatterplot(x=embedded_feats[:,0], y=embedded_feats[:,1], data=df, hue='input_file', s=1)
    fig = scatter.get_figure()
    fig.savefig(f'{args.output}_{p}.png')
    plt.close()

