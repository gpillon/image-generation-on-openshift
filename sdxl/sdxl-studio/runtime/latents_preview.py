import base64
import io

import torch
from PIL import Image

import taesd


def process_latents(diffusers_pipeline, latents):
    """
    Process the given latents to generate a base 64 encoded image.
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