import base64
import io

import torch
from PIL import Image

import taesd


def process_latents(diffusers_pipeline, latents):
    """
    Process the given latents to generate a base 64 encoded image.
    For SDXL models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pipe = diffusers_pipeline.pipeline
    taesd_dec = taesd.Decoder().to(device).requires_grad_(False)
    taesd_dec.load_state_dict(torch.load("taesdxl_decoder.pth", map_location=device, weights_only=True))
    with torch.no_grad():
        decoded = pipe.image_processor.postprocess(taesd_dec(latents.float()).mul_(2).sub_(1))[0]
        # Resize the image to half its size to save on bandwidth
        width, height = decoded.size
        resized_image = decoded.resize((width // 2, height // 2))
        
        img_bytes = io.BytesIO()
        resized_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        image_data = img_bytes.read()
        encoded_image = base64.b64encode(image_data).decode("utf-8")
    
    return encoded_image


def process_flux_latents(flux_pipeline, latents):
    """
    Process the given latents to generate a base 64 encoded image.
    Specialized for Flux models which may have different channel formats.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pipe = flux_pipeline.pipeline
    
    try:
        # Convert latents to images directly using the VAE decoder if possible
        with torch.no_grad():
            # Ensure latents have expected shape for Flux (bsz, channel, height, width)
            if latents.dim() == 4:
                # Standard processing
                images = pipe.vae.decode(latents.to(device=device, dtype=torch.float16)).sample
                images = (images / 2 + 0.5).clamp(0, 1)
                images = images.cpu().permute(0, 2, 3, 1).numpy()
                image = Image.fromarray((images[0] * 255).round().astype("uint8"))
            else:
                # If channels don't match, try reshaping
                # Assuming the latents come in with shape [1, 1, H, W] (single channel)
                print(f"Flux latents shape: {latents.shape}, attempting to adapt")
                # Repeat the channel to get 4 channels if needed
                if latents.shape[1] == 1:
                    latents = latents.repeat(1, 4, 1, 1)
                    print(f"Reshaped to: {latents.shape}")
                images = pipe.vae.decode(latents.to(device=device, dtype=torch.float16)).sample
                images = (images / 2 + 0.5).clamp(0, 1)
                images = images.cpu().permute(0, 2, 3, 1).numpy()
                image = Image.fromarray((images[0] * 255).round().astype("uint8"))
                
            # Resize the image to half its size to save on bandwidth
            width, height = image.size
            resized_image = image.resize((width // 2, height // 2))
            
            img_bytes = io.BytesIO()
            resized_image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            image_data = img_bytes.read()
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            
            return encoded_image
            
    except Exception as e:
        print(f"Error processing Flux latents: {e}")
        # Fallback to a placeholder image if processing fails
        placeholder = Image.new('RGB', (256, 256), color='gray')
        img_bytes = io.BytesIO()
        placeholder.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        image_data = img_bytes.read()
        return base64.b64encode(image_data).decode("utf-8")