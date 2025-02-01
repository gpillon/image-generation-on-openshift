import asyncio
from typing import List, Optional, Tuple
from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str = "ok"


class GenerationRequest(BaseModel):
    prompt: str
    height: Optional[int] = None
    width: Optional[int] = None
    guidance_scale: float = 8.0
    num_inference_steps: int = 50
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    prompt_2: Optional[str] = None
    negative_prompt: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    timesteps: List[int] = None
    sigmas: List[float] = None
    denoising_limit: Optional[float] = 0.8
    eta: float = 0.0
    guidance_rescale: float = 0.0
    original_size: Optional[Tuple[int, int]] = None
    target_size: Optional[Tuple[int, int]] = None
    negative_original_size: Optional[Tuple[int, int]] = None
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0)
    negative_target_size: Optional[Tuple[int, int]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "cat, photo, 4k",
                    "height": 1024,
                    "width": 1024,
                    "guidance_scale": 8.0,
                    "num_inference_steps": 30,
                }
            ]
        }
    }


class GenerationResponse(BaseModel):
    job_id: str


class Job:
    """Represents a generation job."""

    def __init__(self, job_id: str, request: GenerationRequest):
        self.id = job_id
        self.request = request
        self.state = "queued"  # can be 'queued', 'processing', 'completed', or 'error'
        self.result = None  # Will hold the image bytes when completed.
        self.notification_queue: asyncio.Queue = asyncio.Queue()
