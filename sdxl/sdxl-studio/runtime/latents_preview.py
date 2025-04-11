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
            print(f"Flux latents shape: {latents.shape}, dtype: {latents.dtype}, dim: {latents.dim()}")
            
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
                
                # For 3D latents, we'll create a color visualization
                # Reshape to [1, 32, 32, 64] assuming square dimensions
                # This assumes the 1024 can be reshaped to 32x32
                latents_reshaped = latents[0].reshape(32, 32, 64)
                
                # Take the first 3 channels for RGB visualization
                # If we have more channels, we can use PCA or other methods to reduce to 3 channels
                rgb_channels = latents_reshaped[:, :, :3]
                
                # Normalize each channel independently to 0-1 range
                for i in range(3):
                    channel = rgb_channels[:, :, i]
                    min_val = channel.min()
                    max_val = channel.max()
                    if max_val - min_val > 1e-6:
                        rgb_channels[:, :, i] = (channel - min_val) / (max_val - min_val)
                
                # Convert to uint8 and create RGB image
                rgb_image = (rgb_channels * 255).astype(np.uint8)
                image = Image.fromarray(rgb_image)
                
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
        
        # WAN latents should be 5D: [batch, frames, channels, height, width]
        if latents.dim() == 5:
            num_frames = latents.shape[1]
            
            # Select frames to display (ensure we respect the multiple of 4 constraint)
            # For preview, we'll select a subset of frames evenly distributed
            max_preview_frames = min(16, num_frames)  # Show up to 16 frames in preview
            frame_indices = torch.linspace(0, num_frames-1, max_preview_frames).long()
            preview_latents = latents[:, frame_indices]
            
            # Convert to right dtype for VAE
            if hasattr(pipe.vae, 'dtype'):
                vae_dtype = pipe.vae.dtype
            else:
                vae_dtype = torch.float16
                
            # Try to decode the latents using the VAE
            try:
                # Process exactly like the WAN pipeline does
                with torch.no_grad():
                    # Convert to the VAE's expected dtype
                    vae_latents = preview_latents.to(device=device, dtype=vae_dtype)
                    
                    # Apply normalization as in the WAN pipeline
                    if hasattr(pipe.vae.config, 'latents_mean') and hasattr(pipe.vae.config, 'latents_std'):
                        # Apply VAE scaling exactly like in WAN pipeline
                        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
                            1, pipe.vae.config.z_dim, 1, 1, 1
                        ).to(vae_latents.device, vae_latents.dtype)
                        
                        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                            1, pipe.vae.config.z_dim, 1, 1, 1
                        ).to(vae_latents.device, vae_latents.dtype)
                        
                        vae_latents = vae_latents / latents_std + latents_mean
                    
                    # Decode batch of frames
                    # Keep the 5D structure as WAN pipeline expects
                    frames = pipe.vae.decode(vae_latents, return_dict=False)[0]
                    
                    # Process the frames for display
                    if hasattr(pipe, 'video_processor') and hasattr(pipe.video_processor, 'postprocess_video'):
                        # Use the pipeline's own postprocessing if available
                        frames = pipe.video_processor.postprocess_video(frames, output_type='np')
                    else:
                        # Standard normalization
                        frames = (frames / 2 + 0.5).clamp(0, 1)
                        frames = frames.cpu().permute(0, 1, 3, 4, 2).numpy()  # [B, F, H, W, C]
                    
                    # Create a grid from the frames
                    batch_idx = 0  # Use first batch
                    grid_size = int(np.ceil(np.sqrt(len(frame_indices))))
                    
                    # Get dimensions from the first frame
                    frame_height = frames[batch_idx, 0].shape[0]
                    frame_width = frames[batch_idx, 0].shape[1]
                    
                    # Create a single image grid
                    grid_width = grid_size * frame_width
                    grid_height = grid_size * frame_height
                    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                    
                    # Place each frame in the grid
                    for idx in range(len(frame_indices)):
                        if idx >= frames.shape[1]:
                            break
                            
                        row = idx // grid_size
                        col = idx % grid_size
                        
                        y_start = row * frame_height
                        y_end = (row + 1) * frame_height
                        x_start = col * frame_width
                        x_end = (col + 1) * frame_width
                        
                        # Convert to uint8 for PIL
                        frame_img = (frames[batch_idx, idx] * 255).round().astype("uint8")
                        grid_image[y_start:y_end, x_start:x_end] = frame_img
                    
                    # Convert to PIL image
                    grid_pil = Image.fromarray(grid_image)
                    
                    # Add frame numbers
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(grid_pil)
                    for idx in range(len(frame_indices)):
                        if idx >= len(frame_indices):
                            break
                            
                        row = idx // grid_size
                        col = idx % grid_size
                        
                        x = col * frame_width + 5
                        y = row * frame_height + 5
                        
                        # Display actual frame index from original latents
                        frame_num = frame_indices[idx].item()
                        draw.text((x, y), f"F{frame_num}", fill=(255, 255, 255))
                    
            except Exception as e:
                print(f"Failed to decode WAN latents with VAE: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to displaying a latent heatmap grid
                grid_pil = Image.new('RGB', (512, 512), color=(100, 100, 100))
                draw = ImageDraw.Draw(grid_pil)
                draw.text((20, 20), f"Video preview (frames: {num_frames})", fill=(255, 255, 255))
                draw.text((20, 50), f"Error: {str(e)[:60]}...", fill=(255, 255, 255))
        else:
            # For non-standard latent shapes, create an informational image
            grid_pil = Image.new('RGB', (512, 512), color=(100, 150, 200))
            draw = ImageDraw.Draw(grid_pil)
            draw.text((20, 20), f"Unexpected latent shape: {latents.shape}", fill=(255, 255, 255))
            draw.text((20, 50), "Expected 5D tensor [B, F, C, H, W]", fill=(255, 255, 255))
        
        # Resize for bandwidth efficiency
        width, height = grid_pil.size
        resized_image = grid_pil.resize((width // 2, height // 2))
        
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