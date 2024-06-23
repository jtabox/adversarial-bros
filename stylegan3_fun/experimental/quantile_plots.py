# Plot the quantiles of a dataset against each other
# https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot

import matplotlib.pyplot as plt

import os
import sys
from typing import List, Optional, Union, Tuple
import click

from stylegan3_fun import dnnlib
from stylegan3_fun.torch_utils import gen_utils
import copy

import scipy
import numpy as np
import PIL.Image
import torch

from stylegan3_fun import legacy

# Set a predefined style
plt.style.use('ggplot')

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def calculate_quantiles_torch(data, num_quantiles=100):
    quantiles = torch.linspace(0, 1, num_quantiles, device=device)
    return torch.quantile(data, quantiles).cpu().numpy()


# TODO: use 'stylegan3-t', 'stylegan3-r'
cfg = 'stylegan2'
dlatents = []

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)  # TODO: do we need to sample different latents for each?
latents = torch.randn(10000, 512, device=device)

# TODO: separate by number of mapping layers?
# TODO: separate by image resolution
# TODO: see how truncation affects these plots
networks = ['ffhq1024', 'ffhqu1024', 'ffhq512', 'ffhq256', 'ffhqu256']  # ffhq at different resolutions
networks = ['ffhq256', 'celebahq256', 'anime256']  # faces?
networks = ['lsundog256', 'lsuncat256', 'lsunchurch256', 'lsunhorse256']  # lsun
networks = ['afhqcat256', 'cub256', 'sdhorses256', 'sdbicycles256']  # animals

for network_pkl in networks:
    G = gen_utils.load_network('G_ema', network_pkl, cfg, device)
    dlatent = G.mapping(z=latents, c=None, truncation_psi=1.0).detach()
    dlatents.append(dlatent[:, 0])

    del G
    del dlatent

N = len(networks)

quantiles = [calculate_quantiles_torch(data) for data in dlatents]

# Plotting
fig, axs = plt.subplots(N, N, figsize=(4*N, 4*N), constrained_layout=True)
# Set title
fig.suptitle('Quantile-Quantile Plots: $p(\mathcal{W})$', fontsize=16)

for i in range(N):
    for j in range(N):
        if i == j:
            # Leave empty
            axs[i, j].axis('off')

            # data = dlatents[i].cpu().numpy()
            # mean, std = data.mean(), data.std()
            # axs[i, j].hist(data, bins=75, density=True, alpha=0.7, color="skyblue")
            # axs[i, j].axvline(mean, color='darkred', linestyle='dashed', linewidth=2)
            # axs[i, j].set_title(f'Dataset {i+1}\nMean: {mean:.2f}, Std Dev: {std:.2f}')
        else:
            axs[i, j].scatter(quantiles[j], quantiles[i], color="darkblue")
            axs[i, j].plot(quantiles[j], quantiles[j], 'r--', alpha=0.6)  # y=x reference line
            axs[i, j].set_title(f'{networks[i]} vs {networks[j]}')
            axs[i, j].set_xlabel(f'Quantiles for {networks[j]}')
            axs[i, j].set_ylabel(f'Quantiles for {networks[i]}')

# plt.tight_layout()
plt.show()