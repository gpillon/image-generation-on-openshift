import argparse
import os
import sys

# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import parse_args as common_parse_args

def parse_args():
    """Parse arguments with SDXL-specific parameters."""
    parser = argparse.ArgumentParser(description="SDXL Serving Runtime")
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
        "--reload",
        type=bool,
        default=bool(os.getenv("RELOAD", "False").lower() in ("true", "1", "t")),
        help="Enable auto-reload",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("MODEL_ID", "/mnt/models"),
        help="Model ID to load (default: /mnt/models, adapt if you use the refiner model)",
    )
    parser.add_argument(
        "--single-file-model",
        type=str,
        default=os.getenv("SINGLE_FILE_MODEL", None),
        help="Name of a single file model to load",
    )
    parser.add_argument(
        "--use-refiner",
        type=bool,
        default=bool(os.getenv("USE_REFINER", "False").lower() in ("true", "1", "t")),
        help="Use the refiner model",
    )
    parser.add_argument(
        "--refiner-id",
        type=str,
        default=os.getenv("REFINER_ID", None),
        help="Refiner model ID to load (or adapt from /mnt/models)",
    )
    parser.add_argument(
        "--refiner-single-file-model",
        type=str,
        default=os.getenv("REFINER_SINGLE_FILE_MODEL", None),
        help="Name of a single file refiner model to load",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "cuda"),
        help="Device to use, including offloading. Valid values are: 'cuda' (default), 'enable_model_cpu_offload', 'enable_sequential_cpu_offload', 'cpu' (works but unusable...)",
    )
    return parser.parse_args() 