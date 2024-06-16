# Functions used in the main script
import torch
import random
import numpy as np
from PIL import Image

import dnnlib, legacy

def use_last_gen_seed(seed_text):
    seed_text = -1 if seed_text == "" else seed_text.split(" ")[2]
    return int(seed_text)

def generate_image(model, seed, psi):
    # Generate image
    if type(seed) == int and seed >= 0 and seed <= 4294967295:
        seed = seed
    else:
        seed = random.randint(0, 4294967295)
    print(f"Generating image with seed {seed} and psi {psi} using model {model}")
    device = torch.device('cuda')
    # with open(model, 'rb') as f:
    #     G = pickle.load(f)['G_ema']
    with dnnlib.util.open_url(model) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    noise_mode = 'const' # 'const', 'random', 'none'
    translate = (0, 0)
    rotate = 0
    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    img = G(z, label, truncation_psi=psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return [Image.fromarray(img[0].cpu().numpy(), 'RGB'), f"#### Seed:\n## {seed}"]

def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m
