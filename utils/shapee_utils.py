import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh

import numpy as np
def init_from_shapee(prompt, batch_size=1, guidance_scale=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    xm = load_model('transmitter', device=device, cache_dir='/media/data1/zhujh/sd_ckpt/shape-e')
    model = load_model('text300M', device=device, cache_dir='/media/data1/zhujh/sd_ckpt/shape-e')
    diffusion = diffusion_from_config(load_config('diffusion', cache_dir='/media/data1/zhujh/sd_ckpt/shape-e'))

    # batch_size = 1
    # guidance_scale = 15.0
    # prompt = "a shark"

    latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
    t = decode_latent_mesh(xm, latents).tri_mesh()
    rgb = np.zeros_like(t.verts)
    rgb[:,0],rgb[:,1],rgb[:,2] = t.vertex_channels['R'], t.vertex_channels['G'], t.vertex_channels['B']    
    return t.verts, rgb