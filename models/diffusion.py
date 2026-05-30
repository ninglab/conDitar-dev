# =============================================================================
# From: https://github.com/guanjq/targetdiff

# Copyright (c) 2023 Jiaqi Guan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np


def get_beta_schedule(beta_schedule, num_diffusion_timesteps, **kwargs):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    for key in kwargs:
        kwargs[key] = float(kwargs[key])
    
    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    kwargs['beta_start'] ** 0.5,
                    kwargs['beta_end'] ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            kwargs['beta_start'], kwargs['beta_end'], num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        s = dict.get(kwargs, 's', 3)
        betas = np.linspace(-s, s, num_diffusion_timesteps)
        betas = sigmoid(betas) * (kwargs['beta_end'] - kwargs['beta_start']) + kwargs['beta_start']
    elif beta_schedule == "cosine":
        s = dict.get(kwargs, 's', 0.008)
        betas = cosine_beta_schedule(num_diffusion_timesteps, s=s)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)
