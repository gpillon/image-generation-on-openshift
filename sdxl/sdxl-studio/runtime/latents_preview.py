import base64
import io

import torch
from PIL import Image
import numpy as np

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
    
    # Handle the case where flux_pipeline is the direct pipeline object or has a pipeline attribute
    try:
        pipe = flux_pipeline.pipeline
    except (AttributeError, TypeError):
        pipe = flux_pipeline  # Use flux_pipeline directly if it doesn't have .pipeline
    
    try:
        # Convert latents to images directly using the VAE decoder if possible
        with torch.no_grad():
            print(f"Flux latents shape: {latents.shape}, dtype: {latents.dtype}")
            
            # Handle different latent shapes
            if latents.dim() == 4:
                # Standard 4D shape [batch, channel, height, width]
                try:
                    images = pipe.vae.decode(latents.to(device=device, dtype=torch.float16)).sample
                    images = (images / 2 + 0.5).clamp(0, 1)
                    images = images.cpu().permute(0, 2, 3, 1).numpy()
                    image = Image.fromarray((images[0] * 255).round().astype("uint8"))
                except Exception as e:
                    print(f"Error decoding 4D latents: {e}")
                    raise e
                    
            elif latents.dim() == 3:
                # Handle 3D latents with shape [1, 1024, 64]
                print(f"Processing 3D latents with shape {latents.shape}")
                
                # For 3D latents, we need a different approach
                # Typically these are attention maps or internal representations
                # Let's visualize them as a grayscale heatmap
                
                # Take the mean across the embedding dimension to get [1, 1024]
                # Then reshape to a square image
                heatmap = latents.mean(dim=2).detach().cpu().numpy()[0]  # Shape [1024]
                
                # Reshape to a square if possible, or use the closest square
                size = int(heatmap.shape[0] ** 0.5)
                if size * size == heatmap.shape[0]:
                    # Perfect square
                    heatmap_img = heatmap.reshape(size, size)
                else:
                    # Not a perfect square, pad to the next square
                    next_square = (size + 1) ** 2
                    padded = np.zeros(next_square)
                    padded[:heatmap.shape[0]] = heatmap
                    heatmap_img = padded.reshape(size + 1, size + 1)
                
                # Normalize to 0-1 range
                heatmap_img = (heatmap_img - heatmap_img.min()) / (heatmap_img.max() - heatmap_img.min() + 1e-6)
                
                # Convert to PIL Image
                heatmap_img = (heatmap_img * 255).astype(np.uint8)
                image = Image.fromarray(heatmap_img).convert('RGB')
                
                # Resize to a reasonable size
                image = image.resize((512, 512), Image.LANCZOS)
                
                # Add text overlay
                from PIL import ImageDraw
                draw = ImageDraw.Draw(image)
                draw.text((10, 10), f"Progress: Latent shape {latents.shape}", fill=(255, 255, 255))
                
            else:
                print(f"Unexpected tensor dimension: {latents.dim()}")
                placeholder = Image.new('RGB', (512, 512), color=(150, 150, 150))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(placeholder)
                draw.text((10, 10), f"Unknown shape: {latents.shape}", fill=(255, 255, 255))
                image = placeholder
                
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
        import traceback
        traceback.print_exc()
        
        # Fallback to a placeholder image if processing fails
        placeholder = Image.new('RGB', (256, 256), color='gray')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(placeholder)
        draw.text((20, 100), f"Error: {str(e)[:50]}...", fill=(255, 255, 255))
        
        img_bytes = io.BytesIO()
        placeholder.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        image_data = img_bytes.read()
        return base64.b64encode(image_data).decode("utf-8")


def process_wan_latents(wan_pipeline, latents):
    """
    Process the given latents from a WAN model to generate a base64 encoded preview image.
    For WAN text-to-video models, shows intermediate frames during generation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        # Handle the case where wan_pipeline is the direct pipeline object or has a pipeline attribute
        try:
            pipe = wan_pipeline.pipeline
        except (AttributeError, TypeError):
            pipe = wan_pipeline  # Use wan_pipeline directly if it doesn't have .pipeline
        
        print(f"WAN latents shape: {latents.shape}, dtype: {latents.dtype}")
        
        # Instead of trying to decode the latents (which causes type mismatch errors),
        # create a meaningful placeholder image showing generation progress
        placeholder = Image.new('RGB', (512, 512), color=(100, 150, 200))
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(placeholder)
        
        # Draw progress information
        draw.text((20, 20), f"Video generation in progress", fill=(255, 255, 255))
        draw.text((20, 50), f"Shape: {latents.shape}", fill=(255, 255, 255))
        draw.text((20, 80), f"Frames: {latents.shape[1]}", fill=(255, 255, 255))
        
        # Draw a grid representing frames
        if latents.dim() == 5:
            num_frames = latents.shape[1]
            grid_size = min(int(num_frames**0.5) + 1, 10)  # Square grid, max 10x10
            cell_size = 20
            start_x = 20
            start_y = 120
            
            # Draw frame grid
            for i in range(min(num_frames, grid_size * grid_size)):
                row = i // grid_size
                col = i % grid_size
                x = start_x + col * cell_size
                y = start_y + row * cell_size
                draw.rectangle([x, y, x + cell_size - 2, y + cell_size - 2], 
                              fill=(50, 100, 150) if i < latents.shape[1] else (30, 60, 90))
        
        # Resize the image for display
        resized_image = placeholder.resize((256, 256))
        
        img_bytes = io.BytesIO()
        resized_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        image_data = img_bytes.read()
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        
        return encoded_image
    
    except Exception as e:
        print(f"Error processing WAN latents: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to a basic placeholder image if processing fails
        placeholder = Image.new('RGB', (256, 256), color='blue')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(placeholder)
        draw.text((20, 100), f"Video processing: {str(e)[:50]}...", fill=(255, 255, 255))
        
        img_bytes = io.BytesIO()
        placeholder.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        image_data = img_bytes.read()
        return base64.b64encode(image_data).decode("utf-8")