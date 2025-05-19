import torch
import subprocess
import lightning.pytorch as pl

import logging


logger = logging.getLogger(__name__)
def class_fn_from_str(class_str):
    class_module, from_class = class_str.rsplit(".", 1)
    class_module = __import__(class_module, fromlist=[from_class])
    return getattr(class_module, from_class)


class BaseVAE(torch.nn.Module):
    def __init__(self, scale=1.0, shift=0.0):
        super().__init__()
        self.model = torch.nn.Identity()
        self.scale = scale
        self.shift = shift

    def encode(self, x):
        return x/self.scale+self.shift

    def decode(self, x):
        return (x-self.shift)*self.scale


# very bad bugs with nearest sampling
class DownSampleVAE(BaseVAE):
    def __init__(self, down_ratio, scale=1.0, shift=0.0):
        super().__init__()
        self.model = torch.nn.Identity()
        self.scale = scale
        self.shift = shift
        self.down_ratio = down_ratio

    def encode(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=1/self.down_ratio, mode='bicubic', align_corners=False)
        return x/self.scale+self.shift

    def decode(self, x):
         x = (x-self.shift)*self.scale
         x = torch.nn.functional.interpolate(x, scale_factor=self.down_ratio, mode='bicubic', align_corners=False)
         return x


"""
class LatentVAE(BaseVAE):
    def __init__(self, precompute=False, weight_path:str=None):
        super().__init__()
        self.precompute = precompute
        self.model = None
        self.weight_path = weight_path

        from diffusers.models import AutoencoderKL
        setattr(self, "model", AutoencoderKL.from_pretrained(self.weight_path))
        self.scaling_factor = self.model.config.scaling_factor

    @torch.no_grad()
    def encode(self, x):
        assert self.model is not None
        if self.precompute:
            return x.mul_(self.scaling_factor)
        return self.model.encode(x).latent_dist.sample().mul_(self.scaling_factor)

    @torch.no_grad()
    def decode(self, x):
        assert self.model is not None
        return self.model.decode(x.div_(self.scaling_factor)).sample

"""
# In src/models/vae.py

class LatentVAE(BaseVAE):
    def __init__(self, precompute=False, weight_path:str=None):
        super().__init__()
        self.precompute = precompute
        self.model = None
        self.weight_path = weight_path
        print(f"[LatentVAE __init__] precompute set to: {self.precompute}") # Check init value

        from diffusers.models import AutoencoderKL
        # It's good practice to put model loading in a try-except block
        try:
            self.model = AutoencoderKL.from_pretrained(self.weight_path)
            print(f"[LatentVAE __init__] AutoencoderKL model loaded successfully from {self.weight_path}")
            self.scaling_factor = self.model.config.scaling_factor
            print(f"[LatentVAE __init__] Scaling factor: {self.scaling_factor}")
        except Exception as e:
            print(f"[LatentVAE __init__] ERROR loading AutoencoderKL model: {e}")
            raise # Re-raise the exception to stop if model can't load


    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor: # Added type hints for clarity
        print(f"[LatentVAE.encode] Called. self.precompute = {self.precompute}. Input x shape: {x.shape}")
        assert self.model is not None, "VAE model not loaded!"

        if self.precompute:
            print("[LatentVAE.encode] Using precompute path. Multiplying input x by scaling_factor.")
            # This path assumes x is already latents if precompute is True.
            # If x is an image here, this is incorrect for generating new latents.
            output_latents = x.mul(self.scaling_factor) # Use mul not mul_ for safety if x is used elsewhere
            print(f"[LatentVAE.encode] Precompute path output shape: {output_latents.shape}")
            return output_latents
        else:
            print("[LatentVAE.encode] Using standard encode path (AutoencoderKL).")
            latent_dist_obj = self.model.encode(x).latent_dist
            # print(f"[LatentVAE.encode] latent_dist_obj type: {type(latent_dist_obj)}") # For debugging if needed
            
            sampled_latents = latent_dist_obj.sample()
            print(f"[LatentVAE.encode] Shape of sampled_latents from dist.sample(): {sampled_latents.shape}") # Should be [B, 4, H, W]
            
            output_latents = sampled_latents.mul(self.scaling_factor) # Use mul
            print(f"[LatentVAE.encode] Shape of final scaled latents from standard path: {output_latents.shape}")
            return output_latents

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor: # Added type hints
        print(f"[LatentVAE.decode] Called. Input x shape: {x.shape}")
        assert self.model is not None, "VAE model not loaded!"
        
        # Decode the latents
        decoded_output = self.model.decode(x.div(self.scaling_factor)) # x here are latents
        
        # Handle different output types from diffusers models
        if hasattr(decoded_output, 'sample'):
            # DecoderOutput object from diffusers has a 'sample' attribute
            print(f"[LatentVAE.decode] Found DecoderOutput with sample attribute")
            image_tensor = decoded_output.sample
            print(f"[LatentVAE.decode] Shape of decoded image tensor from sample attribute: {image_tensor.shape}")
            return image_tensor
        elif isinstance(decoded_output, torch.Tensor):
            # This is if self.model.decode directly returns the image tensor
            print(f"[LatentVAE.decode] Shape of decoded image tensor: {decoded_output.shape}")
            return decoded_output
        else:
            print(f"[LatentVAE.decode] ERROR: Unexpected output type from model.decode: {type(decoded_output)}")
            # Try to extract the sample in different ways
            if hasattr(decoded_output, "images"):
                return decoded_output.images
            # If all else fails
            return None

def uint82fp(x):
    x = x.to(torch.float32)
    x = (x - 127.5) / 127.5
    return x

def fp2uint8(x):
    x = torch.clip_((x + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)
    return x

