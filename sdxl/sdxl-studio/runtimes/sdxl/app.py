import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
import os
from PIL import Image

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import sys
import torch

# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.app_base import BaseApp
from common.classes import GenerationRequest, ModelCapabilities, Job
from common.logging_config import logging_config
from common.watermark import add_watermark

# Import model-specific helpers
from helpers import parse_args

# Import SDXL-specific modules
from taesd import Decoder
import taesd

# Import model implementation
from diffusers_model import DiffusersPipeline

# Set up logging
logging_config()
_log = logging.getLogger(__name__)


# Custom latents processing for SDXL
def process_latents(diffusers_pipeline, latents):
    """
    Process the given latents to generate a base 64 encoded image.
    For SDXL models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pipe = diffusers_pipeline.pipeline
    taesd_dec = Decoder().to(device).requires_grad_(False)
    taesd_dec.load_state_dict(torch.load("sdxl/taesdxl_decoder.pth", map_location=device, weights_only=True))
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model and start background queue processing"""
    # Start the background queue processor
    queue_task = asyncio.create_task(sdxl_app.process_queue())

    yield

    # Cancel the background queue processor on shutdown
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        _log.info("Queue processor cancelled.")


class SDXLApp(BaseApp):
    def __init__(self, args):
        self.generation_workers = args.generation_workers
        super().__init__("SDXL", lifespan_func=lifespan)
        
        # Initialize model
        self.pipeline = DiffusersPipeline(args)
        self.use_watermark = os.getenv("USE_WATERMARK", "False").lower() in ("true", "1", "t")
        self.watermark_text = os.getenv("WATERMARK_TEXT", "SDXL Studio")
        
    async def process_queue(self):
        """Process queue of image generation jobs using workers."""
        from common.app_base import jobs, job_queue, queue_list
        
        _log.info(f"Starting background queue processing with {self.generation_workers} workers")

        # Load model at startup
        try:
            _log.info("Preloading SDXL model")
            self.pipeline.load()
            _log.info("SDXL model preloaded successfully")
        except Exception as e:
            _log.error(f"Error preloading SDXL model: {e}")
            import traceback
            _log.error(traceback.format_exc())

        # Set up worker tasks
        worker_tasks = []
        for i in range(self.generation_workers):
            task = asyncio.create_task(self.worker(i, job_queue, self.pipeline))
            worker_tasks.append(task)

        # Wait for the worker tasks to complete (they should run indefinitely)
        await asyncio.gather(*worker_tasks)

    async def worker(self, worker_id, job_queue, pipeline_instance):
        """
        Worker function that processes jobs from the queue.
        Callback functions are used by predict to notify the client of progress.
        """
        from common.app_base import jobs, queue_list
        
        _log.info(f"Worker {worker_id} started")

        while True:
            # Get a job from the queue
            job = await job_queue.get()

            try:
                _log.info(f"Worker {worker_id} processing job {job.id}")
                jobs[job.id].state = "processing"

                # Remove this job from the queue list to update queue positions
                if job.id in queue_list:
                    queue_list.remove(job.id)
                await self.notify_all_queue_positions()

                # Get the current event loop for callbacks to use
                loop = asyncio.get_running_loop()

                # Set up callback functions for progress reporting
                def callback_func_base(_pipe, step, _timestep, callback_kwargs):
                    if step % 5 == 0 or step == 1:
                        # Calculate progress percentage
                        total_steps = job.request.num_inference_steps
                        progress = int((step / total_steps) * 50)  # Up to 50% for base model

                        # Send progress update
                        msg = {
                            "status": "progress",
                            "progress": progress,
                            "step": step,
                            "total_steps": total_steps,
                            "job_id": job.id,
                        }

                        # If latents are available in the callback, process and send them
                        if "latents" in callback_kwargs:
                            try:
                                latents = callback_kwargs["latents"]
                                latent_img = process_latents(pipeline_instance, latents)
                                msg["image"] = latent_img
                            except Exception as e:
                                _log.error(f"Error processing latents: {e}")

                        # Add to the notification queue using the captured loop
                        future = asyncio.run_coroutine_threadsafe(
                            job.notification_queue.put(msg), loop
                        )
                        future.result()  # Wait for completion to ensure proper ordering
                    return callback_kwargs

                def callback_func_refiner(_pipe, step, _timestep, callback_kwargs):
                    if step % 5 == 0 or step == 1:
                        # Calculate progress percentage for refiner (50-100%)
                        total_steps = job.request.num_inference_steps
                        base_progress = 50  # Base model already did 50%
                        refiner_progress = int((step / total_steps) * 50)
                        progress = base_progress + refiner_progress

                        # Send progress update
                        msg = {
                            "status": "refining",
                            "progress": progress,
                            "step": step,
                            "total_steps": total_steps,
                            "job_id": job.id,
                        }

                        # Add to the notification queue using the captured loop
                        future = asyncio.run_coroutine_threadsafe(
                            job.notification_queue.put(msg), loop
                        )
                        future.result()  # Wait for completion to ensure proper ordering
                    return callback_kwargs

                # Process the image - model should already be loaded
                if not pipeline_instance.ready:
                    pipeline_instance.load()
                
                # Generate the image
                pil_image = await asyncio.to_thread(
                    pipeline_instance.predict,
                    job.request,
                    callback_func_base,
                    callback_func_refiner,
                )

                # Convert the PIL image to base64 for sending over JSON
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Apply watermark if configured
                if self.use_watermark:
                    _log.info(f"Applying watermark with text: {self.watermark_text}")
                    img_str = add_watermark(img_str, self.watermark_text)

                # Update the job with the result
                job.result = img_str
                job.state = "completed"

                # Notify clients of completion
                completion_msg = {"status": "completed", "image": img_str, "job_id": job.id}
                await job.notification_queue.put(completion_msg)

                _log.info(f"Job {job.id} completed successfully")

            except Exception as e:
                _log.error(f"Error processing job {job.id}: {e}")
                import traceback
                _log.error(traceback.format_exc())

                # Update job state and notify clients of error
                job.state = "error"
                error_msg = {"status": "error", "message": str(e), "job_id": job.id}
                await job.notification_queue.put(error_msg)

            finally:
                # Mark the task as done
                job_queue.task_done()
    
    def get_capabilities(self) -> ModelCapabilities:
        """Return the capabilities of the SDXL model."""
        return ModelCapabilities(
            model_name="SDXL",
            supported_parameters={
                "prompt": "string",
                "height": "integer",
                "width": "integer", 
                "negative_prompt": "string",
                "guidance_scale": "float",
                "num_inference_steps": "integer",
                "seed": "integer"
            },
            parameter_ranges={
                "height": {"min": 512, "max": 1024, "step": 64},
                "width": {"min": 512, "max": 1024, "step": 64},
                "guidance_scale": {"min": 1.0, "max": 15.0, "step": 0.1},
                "num_inference_steps": {"min": 20, "max": 100, "step": 1}
            },
            default_values={
                "height": 1024,
                "width": 1024,
                "guidance_scale": 8.0,
                "num_inference_steps": 50
            }
        )

    async def get_video(self, job_id: str):
        """
        GET endpoint to serve the generated video file for a specific job.
        Returns the video file as a streaming response.
        """
        video_path = f"/tmp/temp_output.mp4"
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        def iterfile():
            with open(video_path, mode="rb") as file_like:
                yield from file_like
        
        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=video_{job_id}.mp4"}
        )


if __name__ == "__main__":
    import uvicorn
    
    args = parse_args()
    sdxl_app = SDXLApp(args)
    
    _log.info(f"Starting SDXL service on port {args.port}")
    uvicorn.run(
        sdxl_app.app,
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
    ) 