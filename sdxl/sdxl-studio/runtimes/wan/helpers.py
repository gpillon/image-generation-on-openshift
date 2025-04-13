import sys
import os

# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import BaseArgumentParser


class WanArgumentParser(BaseArgumentParser):
    """WAN-specific argument parser extending the base parser."""
    
    def __init__(self):
        super().__init__(description="WAN Model API Server")
        self._add_wan_arguments()
    
    def _add_wan_arguments(self):
        """Add WAN-specific arguments."""
        # Override model-id with WAN-specific default
        self.parser.add_argument(
            "--model-id", 
            type=str, 
            default=os.getenv("MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            help="Path to the HuggingFace model ID"
        )
        self.parser.add_argument(
            "--safetensor-fast-load",
            type=bool,
            default=bool(os.getenv("SAFETENSOR_FAST_LOAD", "True").lower() in ("true", "1", "t")),
            help="Enable fast loading of safetensors",
        )
        
        # Video generation parameters
        self.parser.add_argument(
            "--fps", 
            type=int, 
            default=int(os.getenv("FPS", "15")),
            help="Frames per second for video output"
        )
        self.parser.add_argument(
            "--num-frames", 
            type=int, 
            default=int(os.getenv("NUM_FRAMES", "80")),
            help="Default number of frames to generate"
        )


def parse_args():
    """Parse arguments with WAN-specific parameters."""
    parser = WanArgumentParser()
    return parser.parse_args() 