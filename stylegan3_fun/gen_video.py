# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import copy
import os
import re
from typing import List, Optional, Tuple, Union

import click
from stylegan3_fun import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm

from stylegan3_fun import legacy
from stylegan3_fun.torch_utils import gen_utils


# ----------------------------------------------------------------------------


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


# ----------------------------------------------------------------------------


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


# ----------------------------------------------------------------------------


def make_transform(translate: Tuple[float,float], angle: float):
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


# ----------------------------------------------------------------------------


def gen_interp_video(G,
                     mp4: str,
                     seeds: List[int],
                     shuffle_seed: int = None,
                     w_frames: int = 60*4,
                     kind: str = 'cubic',
                     grid_dims: Tuple[int] = (1,1),
                     num_keyframes: int = None,
                     wraps: int = 2,
                     psi: float = 1.0,
                     device: torch.device = torch.device('cuda'),
                     stabilize_video: bool = True,
                     **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if stabilize_video:
        # Thanks to @RiversHaveWings and @nshepperd1
        if hasattr(G.synthesis, 'input'):
            shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0))
            G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
            G.synthesis.input.affine.weight.data.zero_()

    # Get the Generator's transform
    m = G.synthesis.input.transform if hasattr(G.synthesis, 'input') else None

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    ws = G.mapping(z=zs, c=None, truncation_psi=psi)
    _ = G.synthesis(ws[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        # Construct an inverse affine matrix and pass to the generator. The generator expects 
        # this matrix as an inverse to avoid potentially failing numerical operations in the network.
        if hasattr(G.synthesis, 'input'):
            # Set default values for each affine transformation
            total_rotation = 0.0  # If >= 0.0, will rotate the pixels counter-clockwise w.r.t. the center; in radians
            total_translation_x = 0.0  # If >= 0.0, will translate all pixels to the right; if <= 0.0, to the left
            total_translation_y = 0.0  # If >= 0.0, will translate all pixels upwards; if <= 0.0, downwards
            total_scale_x = 1.0  # If <= 1.0, will zoom in; else, will zoom out (x-axis)
            total_scale_y = 1.0  # If <= 1.0, will zoom in; else, will zoom out (y-axis)
            total_shear_x = 0.0  # If >= 0.0, will shear pixels to the right, keeping y fixed; if <= 0.0, to the left
            total_shear_y = 0.0  # If >= 0.0, will shear pixels upwards, keeping x fixed; if <= 0.0, downwards
            mirror_x = False  # Mirror along the x-axis; if True, will flip the image horizontally (can't be a function of frame_idx)
            mirror_y = False  # Mirror along the y-axis; if True, will flip the image vertically (can't be a function of frame_idx)

            # Go nuts with these. They can be constants as above to fix centering/rotation in your video, 
            # or you can make them functions of frame_idx to animate them, such as (uncomment as many as you want to try):
            # total_scale_x = 1 + np.sin(np.pi*frame_idx/(num_keyframes * w_frames))/2  # will oscillate between 0.5 and 1.5
            # total_rotation = 4*np.pi*frame_idx/(num_keyframes * w_frames)  # 4 will dictate the number of rotations, so 1 full rotation
            # total_shear_y = 2*np.sin(2*np.pi*frame_idx/(num_keyframes * w_frames))  # will oscillate between -2 and 2

            # We then use these values to construct the affine matrix
            m = gen_utils.make_affine_transform(m, angle=total_rotation, translate_x=total_translation_x,
                                                translate_y=total_translation_y, scale_x=total_scale_x,
                                                scale_y=total_scale_y, shear_x=total_shear_x, shear_y=total_shear_y,
                                                mirror_x=mirror_x, mirror_y=mirror_y)
            m = np.linalg.inv(m)
            # Finally, we pass the matrix to the generator
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # The rest stays the same, for all you gen_video.py lovers out there
        for yi in range(grid_h):
            for xi in range(grid_w):
                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                img = G.synthesis(ws=w.unsqueeze(0), noise_mode='const')[0]
                imgs.append(img)
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()


# ----------------------------------------------------------------------------


def parse_range(s: Union[str, List[int]]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    """
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    """Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--stabilize-video', is_flag=True, help='Stabilize the video by anchoring the mapping to w_avg')
@click.option('--output', help='Output .mp4 filename', type=str, required=True, metavar='FILE')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    stabilize_video: bool,
    w_frames: int,
    output: str
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    gen_interp_video(G=G, mp4=output, bitrate='12M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames,
                     seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, stabilize_video=stabilize_video)


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter


# ----------------------------------------------------------------------------
