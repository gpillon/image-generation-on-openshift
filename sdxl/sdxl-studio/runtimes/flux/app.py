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
from common.latents_preview import LatentsProcessor
from common.watermark import add_watermark

# Import model-specific helpers
from helpers import parse_args
from latents import FluxLatentsProcessor

# Import model implementation
from flux_model import FluxModelPipeline

# Set up logging
logging_config()
_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model and start background queue processing"""
    # Start the background queue processor
    queue_task = asyncio.create_task(flux_app.process_queue())

    yield

    # Cancel the background queue processor on shutdown
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        _log.info("Queue processor cancelled.")


class FluxApp(BaseApp):
    def __init__(self, args):
        self.generation_workers = args.generation_workers
        super().__init__("Flux", lifespan_func=lifespan)
        
        # Initialize model
        self.pipeline = FluxModelPipeline(args)
        self.use_watermark = os.getenv("USE_WATERMARK", "False").lower() in ("true", "1", "t")
        self.watermark_text = os.getenv("WATERMARK_TEXT", "SDXL Studio Flux")
        self.latents_processor = FluxLatentsProcessor()
        
    async def process_queue(self):
        """Process queue of image generation jobs using workers."""
        from common.app_base import jobs, job_queue, queue_list
        
        _log.info(f"Starting background queue processing with {self.generation_workers} workers")
         
        # Load model at startup
        try:
            _log.info("Preloading Flux model")
            self.pipeline.load()
            _log.info("Flux model preloaded successfully")
        except Exception as e:
            _log.error(f"Error preloading Flux model: {e}")
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
                        progress = int((step / total_steps) * 100)

                        # Send progress update
                        msg = {
                            "status": "processing",
                            "progress": progress,
                            "step": step,
                            "total_steps": total_steps,
                        }

                        # If latents are available in the callback, process and send them
                        if "latents" in callback_kwargs:
                            try:
                                latents = callback_kwargs["latents"]
                                latent_img = self.latents_processor.process_latents(pipeline_instance, latents)
                                msg["latent_image"] = latent_img
                            except Exception as e:
                                _log.error(f"Error processing latents: {e}")

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
                completion_msg = {"status": "completed", "image": img_str}
                await job.notification_queue.put(completion_msg)

                _log.info(f"Job {job.id} completed successfully")

            except Exception as e:
                _log.error(f"Error processing job {job.id}: {e}")
                import traceback
                _log.error(traceback.format_exc())

                # Update job state and notify clients of error
                job.state = "error"
                error_msg = {"status": "error", "message": str(e)}
                await job.notification_queue.put(error_msg)

            finally:
                # Mark the task as done
                job_queue.task_done()
    
    def get_capabilities(self) -> ModelCapabilities:
        """Return the capabilities of the Flux model."""
        return ModelCapabilities(
            model_name="Flux",
            supported_parameters={
                "prompt": "string",
                "height": "integer",
                "width": "integer", 
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
                "guidance_scale": 5.0,
                "num_inference_steps": 50
            }
        )


if __name__ == "__main__":
    import uvicorn
    
    args = parse_args()
    flux_app = FluxApp(args)
    
    _log.info(f"Starting Flux service on port {args.port}")
    uvicorn.run(
        flux_app.app,
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
    ) 