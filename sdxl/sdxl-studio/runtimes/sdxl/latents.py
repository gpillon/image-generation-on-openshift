import os
import sys
import torch
from PIL import Image

# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.latents_preview import LatentsProcessor

try:
    from taesd import Decoder
except ImportError:
    # Fallback import path
    from .taesd import Decoder


class SDXLLatentsProcessor(LatentsProcessor):
    """Implementation of LatentsProcessor for SDXL models."""
    
    def process_latents(self, pipeline, latents):
        """
        Process the given latents to generate a PIL image.
        For SDXL models, uses the TAESD decoder.
        
        Args:
            pipeline: The SDXL pipeline
            latents: The latent representation
            
        Returns:
            PIL Image generated from the latents
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Handle the case where pipeline is the direct pipeline object or has a pipeline attribute
        try:
            pipe = pipeline.pipeline
        except (AttributeError, TypeError):
            pipe = pipeline  # Use pipeline directly if it doesn't have .pipeline
        
        # Try to load the TAESD decoder from the expected location
        try:
            taesd_dec = Decoder().to(device).requires_grad_(False)
            
            # First try to load from the current directory
            decoder_path = "sdxl/taesdxl_decoder.pth"
            if not os.path.exists(decoder_path):
                # Try with path relative to the file
                decoder_path = os.path.join(os.path.dirname(__file__), "taesdxl_decoder.pth")
            
            taesd_dec.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
            
            with torch.no_grad():
                decoded = pipe.image_processor.postprocess(taesd_dec(latents.float()).mul_(2).sub_(1))[0]
                return decoded
                
        except Exception as e:
            # Fallback: Use VAE if available or simple normalization
            try:
                if hasattr(pipe, "vae") and hasattr(pipe.vae, "decode"):
                    # Use VAE to decode
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
            
            except Exception as fallback_error:
                # Return a placeholder image if all else fails
                img = Image.new('RGB', (512, 512), color=(200, 200, 200))
                return img 