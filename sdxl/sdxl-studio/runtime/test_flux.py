#!/usr/bin/env python3
"""
Test script for the Flux pipeline

Usage:
    python test_flux.py

This will test loading the Flux model and generating a simple image.
"""

import os
import logging
import argparse
from datetime import datetime

import torch
from flux_model import FluxModelPipeline
from helpers import logging_config

# Set up logging
logging_config()
_log = logging.getLogger(__name__)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the Flux pipeline")
    parser.add_argument("--device", type=str, default="enable_model_cpu_offload", help="Device to use")
    parser.add_argument("--prompt", type=str, default="A cat wearing a red fedora holding a sign that says hello RedHat", help="Prompt for image generation")
    args = parser.parse_args()
    
    # Check for HUGGINGFACE_TOKEN
    if not os.getenv("HUGGINGFACE_TOKEN"):
        _log.warning("HUGGINGFACE_TOKEN environment variable not set. Some models may not be accessible.")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Create and load the model
    _log.info(f"Initializing Flux pipeline with device={args.device}")
    flux_pipeline = FluxModelPipeline(args)
    flux_pipeline.load()
    
    # Generate an image
    _log.info(f"Generating image with prompt: '{args.prompt}'")
    
    def callback_func(pipe, step, timestep, callback_kwargs):
        _log.info(f"Step {step}/{pipe.num_inference_steps}")
        return callback_kwargs
    
    image = flux_pipeline.predict(
        argparse.Namespace(
            prompt=args.prompt,
            negative_prompt="",
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=3.5,
            seed=42
        ),
        callback_func
    )
    
    # Save the image
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"output/flux_test-{timestamp}.png"
    image.save(filename)
    
    _log.info(f"Image saved to {filename}")
    _log.info("Test completed successfully!") 