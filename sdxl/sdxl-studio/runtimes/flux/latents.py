import os
import sys
import torch
import numpy as np
from PIL import Image

# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.latents_preview import LatentsProcessor


class FluxLatentsProcessor(LatentsProcessor):
    """Implementation of LatentsProcessor for Flux models."""
    
    def process_latents(self, pipeline, latents):
        """
        Process latents from a Flux model to generate a preview image.
        
        Args:
            pipeline: The Flux model pipeline
            latents: The latent representation
            
        Returns:
            PIL Image generated from the latents
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        try:
            # Handle the case where pipeline is the direct pipeline object or has a pipeline attribute
            try:
                pipe = pipeline.pipeline
            except (AttributeError, TypeError):
                pipe = pipeline  # Use pipeline directly if it doesn't have .pipeline
            
            # Convert latents to images directly using the VAE decoder if possible
            if latents.dim() == 4:
                # Standard 4D shape [batch, channel, height, width]
                try:
                    if hasattr(pipe, "vae") and hasattr(pipe.vae, "decode"):
                        # Use the VAE to decode latents
                        images = pipe.vae.decode(latents.to(device=device, dtype=torch.float16)).sample
                        images = (images / 2 + 0.5).clamp(0, 1)
                        images = images.cpu().permute(0, 2, 3, 1).numpy()
                        return Image.fromarray((images[0] * 255).round().astype("uint8"))
                    else:
                        # Simple normalization fallback
                        img = latents[0, :3].detach().cpu().permute(1, 2, 0).numpy()
                        # Normalize channels
                        for c in range(3):
                            c_min, c_max = img[:, :, c].min(), img[:, :, c].max()
                            if c_max > c_min:
                                img[:, :, c] = (img[:, :, c] - c_min) / (c_max - c_min)
                        return Image.fromarray((img * 255).round().astype("uint8"))
                except Exception as e:
                    # Simple normalization fallback
                    img = latents[0, :3].detach().cpu().permute(1, 2, 0).numpy()
                    # Normalize channels
                    for c in range(3):
                        c_min, c_max = img[:, :, c].min(), img[:, :, c].max()
                        if c_max > c_min:
                            img[:, :, c] = (img[:, :, c] - c_min) / (c_max - c_min)
                    return Image.fromarray((img * 255).round().astype("uint8"))
                    
            elif latents.dim() == 3:
                # Handle 3D latents with shape [1, H*W, C]
                # Get total pixels and channels
                total_pixels = latents.shape[1]
                num_channels = latents.shape[2]
                
                # Calculate dimensions that preserve the original aspect ratio
                # Start with a square approximation
                base_dim = int(np.sqrt(total_pixels))
                
                # Find the closest dimensions that multiply to total_pixels
                height = base_dim
                while total_pixels % height != 0 and height > 1:
                    height -= 1
                width = total_pixels // height
                
                # Reshape to [height, width, channels]
                latents_reshaped = latents[0].reshape(height, width, num_channels)
                
                # Take the first 3 channels for RGB visualization
                if num_channels >= 3:
                    rgb_channels = latents_reshaped[:, :, :3]
                else:
                    # If less than 3 channels, repeat the last channel
                    rgb_channels = torch.zeros(height, width, 3)
                    for i in range(min(3, num_channels)):
                        rgb_channels[:, :, i] = latents_reshaped[:, :, i]
                    for i in range(min(3, num_channels), 3):
                        rgb_channels[:, :, i] = latents_reshaped[:, :, num_channels-1]
                
                # Convert to numpy array and normalize each channel independently to 0-1 range
                rgb_channels = rgb_channels.detach().cpu().numpy()
                
                # Normalize each channel
                for c in range(3):
                    c_min, c_max = rgb_channels[:, :, c].min(), rgb_channels[:, :, c].max()
                    if c_max > c_min:
                        rgb_channels[:, :, c] = (rgb_channels[:, :, c] - c_min) / (c_max - c_min)
                
                # Convert to uint8 and create PIL image
                rgb_uint8 = (rgb_channels * 255).round().astype(np.uint8)
                return Image.fromarray(rgb_uint8)
            
            # If we get here, we don't know how to handle the latents
            return Image.new('RGB', (512, 512), color=(150, 150, 150))
        
        except Exception as e:
            # Return a placeholder image on error
            img = Image.new('RGB', (512, 512), color=(200, 100, 100))
            return img 