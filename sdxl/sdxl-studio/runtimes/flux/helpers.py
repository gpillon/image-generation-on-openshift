import argparse
import os
import sys

# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import parse_args as common_parse_args

def parse_args():
    """Parse arguments with Flux-specific parameters."""
    parser = argparse.ArgumentParser(description="Flux Serving Runtime")
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=int(os.getenv("GENERATION_WORKERS", "1")),
        help="Number of image processing workers",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8080")),
        help="Port number to run the server on",
    )
    parser.add_argument(
        "--safetensor-fast-load",
        type=bool,
        default=bool(os.getenv("SAFETENSOR_FAST_LOAD", "True").lower() in ("true", "1", "t")),
        help="Enable fast loading of safetensors",
    )
    parser.add_argument(
        "--reload",
        type=bool,
        default=bool(os.getenv("RELOAD", "False").lower() in ("true", "1", "t")),
        help="Enable auto-reload",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("MODEL_ID", "/mnt/models"),
        help="Model ID to load",
    )
    parser.add_argument(
        "--single-file-model",
        type=str,
        default=os.getenv("SINGLE_FILE_MODEL", None),
        help="Name of a single file model to load",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.getenv("REPO_ID", "black-forest-labs/FLUX.1-schnell"),
        help="Flux model repo ID for Hugging Face",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "cuda"),
        help="Device to use, including offloading. Valid values are: 'cuda' (default), 'cpu'",
    )
    return parser.parse_args() 