import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.latents_preview import LatentsProcessor


class WanLatentsProcessor(LatentsProcessor):
    """Implementation of LatentsProcessor for WAN text-to-video models."""
    
    def process_latents(self, pipeline, latents):
        """
        Process latents from a WAN model to generate a preview image.
        For WAN text-to-video models, shows intermediate frames during generation.
        
        Args:
            pipeline: The WAN model pipeline
            latents: The latent representation (5D tensor for video)
            
        Returns:
            PIL Image showing a grid of video frames
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        try:
            # Handle the case where pipeline is the direct pipeline object or has a pipeline attribute
            try:
                pipe = pipeline.pipeline
            except (AttributeError, TypeError):
                pipe = pipeline  # Use pipeline directly if it doesn't have .pipeline
            
            # WAN latents should be 5D: [batch, frames, channels, height, width]
            if latents.dim() == 5:
                batch_size, num_frames, num_channels, height, width = latents.shape
                
                # For preview, we'll create a grid of frames
                # Determine grid size based on number of frames
                import math
                grid_size = min(4, math.ceil(math.sqrt(num_frames)))
                
                # Choose frames to display
                if num_frames <= 16:
                    # Use all frames if <= 16
                    frame_indices = list(range(num_frames))
                else:
                    # Sample frames evenly
                    frame_indices = [
                        int(i * (num_frames - 1) / (grid_size * grid_size - 1))
                        for i in range(grid_size * grid_size)
                    ]
                
                # For the first batch element only
                batch_idx = 0
                
                # Decode the frames using the model's VAE
                try:
                    # If we have a VAE decoder, use it
                    if hasattr(pipe, "vae") and hasattr(pipe.vae, "decode"):
                        # Prepare selected frames for decoding
                        selected_latents = latents[batch_idx, frame_indices]
                        
                        # Reshape to [n_frames, channels, height, width]
                        selected_latents = selected_latents.to(device=device, dtype=torch.float16)
                        
                        # Decode frames 4 at a time to avoid OOM
                        max_batch = 4
                        decoded_frames = []
                        
                        for i in range(0, len(frame_indices), max_batch):
                            batch_latents = selected_latents[i:i+max_batch]
                            
                            # Decode the latents
                            decoded_batch = pipe.vae.decode(batch_latents).sample
                            
                            # Process to image format (0-1 range)
                            decoded_batch = (decoded_batch / 2 + 0.5).clamp(0, 1)
                            
                            # Convert to numpy
                            decoded_batch = decoded_batch.cpu().permute(0, 2, 3, 1).numpy()
                            
                            decoded_frames.append(decoded_batch)
                        
                        # Combine the batches
                        frames = np.concatenate(decoded_frames, axis=0)
                        
                    else:
                        # Use a simpler approach if no VAE
                        # Just normalize the latents and treat the first 3 channels as RGB
                        frames = []
                        for idx in frame_indices:
                            frame = latents[batch_idx, idx, :3].detach().cpu().permute(1, 2, 0).numpy()
                            
                            # Normalize each channel independently to 0-1 range
                            for c in range(3):
                                c_min, c_max = frame[:, :, c].min(), frame[:, :, c].max()
                                if c_max > c_min:
                                    frame[:, :, c] = (frame[:, :, c] - c_min) / (c_max - c_min)
                            
                            frames.append(frame)
                        
                        # Stack the frames
                        frames = np.stack(frames)
                
                except Exception as e:
                    # Fallback: just normalize the latents and show the first 3 channels
                    frames = []
                    for idx in frame_indices:
                        frame = latents[batch_idx, idx, :3].detach().cpu().permute(1, 2, 0).numpy()
                        
                        # Normalize each channel independently to 0-1 range
                        for c in range(3):
                            c_min, c_max = frame[:, :, c].min(), frame[:, :, c].max()
                            if c_max > c_min:
                                frame[:, :, c] = (frame[:, :, c] - c_min) / (c_max - c_min)
                        
                        frames.append(frame)
                    
                    # Stack the frames
                    frames = np.stack(frames)
                
                # Get dimensions from the first frame
                frame_height = frames[0].shape[0]
                frame_width = frames[0].shape[1]
                
                # Create a single image grid
                grid_width = grid_size * frame_width
                grid_height = grid_size * frame_height
                grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                
                # Place each frame in the grid
                for idx, frame_idx in enumerate(frame_indices):
                    if idx >= len(frames):
                        break
                        
                    row = idx // grid_size
                    col = idx % grid_size
                    
                    y_start = row * frame_height
                    y_end = (row + 1) * frame_height
                    x_start = col * frame_width
                    x_end = (col + 1) * frame_width
                    
                    # Convert to uint8 for PIL
                    frame_img = (frames[idx] * 255).round().astype("uint8")
                    grid_image[y_start:y_end, x_start:x_end] = frame_img
                
                # Convert to PIL image
                grid_pil = Image.fromarray(grid_image)
                
                # Add frame numbers
                draw = ImageDraw.Draw(grid_pil)
                for idx, frame_idx in enumerate(frame_indices):
                    if idx >= len(frames):
                        break
                    
                    row = idx // grid_size
                    col = idx % grid_size
                    
                    x = col * frame_width + 5
                    y = row * frame_height + 5
                    
                    # Draw frame number
                    draw.text((x, y), f"F{frame_idx}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                
                return grid_pil
            
            # Handle standard 4D latents if that's what we get
            elif latents.dim() == 4:
                # Standard shape [batch, channel, height, width] - handle like a SDXL model
                try:
                    if hasattr(pipe, "vae") and hasattr(pipe.vae, "decode"):
                        # Use VAE to decode
                        images = pipe.vae.decode(latents.to(device=device, dtype=torch.float16)).sample
                        images = (images / 2 + 0.5).clamp(0, 1)
                        images = images.cpu().permute(0, 2, 3, 1).numpy()
                        # Convert to PIL
                        return Image.fromarray((images[0] * 255).round().astype("uint8"))
                    else:
                        # Simple normalization fallback
                        img = latents[0, :3].detach().cpu().permute(1, 2, 0).numpy()
                        # Normalize channels
                        for c in range(3):
                            c_min, c_max = img[:, :, c].min(), img[:, :, c].max()
                            if c_max > c_min:
                                img[:, :, c] = (img[:, :, c] - c_min) / (c_max - c_min)
                        # Convert to PIL
                        return Image.fromarray((img * 255).round().astype("uint8"))
                
                except Exception as e:
                    # Simple normalization fallback
                    img = latents[0, :3].detach().cpu().permute(1, 2, 0).numpy()
                    # Normalize channels
                    for c in range(3):
                        c_min, c_max = img[:, :, c].min(), img[:, :, c].max()
                        if c_max > c_min:
                            img[:, :, c] = (img[:, :, c] - c_min) / (c_max - c_min)
                    # Convert to PIL
                    return Image.fromarray((img * 255).round().astype("uint8"))
            
            # If we get here, we don't know how to handle the latents
            return Image.new('RGB', (256, 256), color='grey')
        
        except Exception as e:
            # Return a placeholder image on error
            img = Image.new('RGB', (256, 256), color='red')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Error: {str(e)}", fill=(255, 255, 255))
            return img 