import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
import os
from PIL import Image, ImageDraw

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

# Import model-specific modules
from latents import WanLatentsProcessor
from helpers import parse_args

# Import model implementation
from wan_model import WanModelPipeline

# Set up logging
logging_config()
_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model and start background queue processing"""
    # Start the background queue processor
    queue_task = asyncio.create_task(wan_app.process_queue())

    yield

    # Cancel the background queue processor on shutdown
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        _log.info("Queue processor cancelled.")


class WanApp(BaseApp):
    def __init__(self, args):
        self.generation_workers = args.generation_workers
        super().__init__("WAN", lifespan_func=lifespan)
        
        # Initialize model
        self.pipeline = WanModelPipeline(args)
        self.use_watermark = os.getenv("USE_WATERMARK", "False").lower() in ("true", "1", "t")
        self.watermark_text = os.getenv("WATERMARK_TEXT", "SDXL Studio WAN")
        self.latents_processor = WanLatentsProcessor()
        
    async def process_queue(self):
        """Process queue of image generation jobs using workers."""
        from common.app_base import jobs, job_queue, queue_list
        
        _log.info(f"Starting background queue processing with {self.generation_workers} workers")

        # Load model at startup
        try:
            _log.info("Preloading WAN model")
            self.pipeline.load()
            _log.info("WAN model preloaded successfully")
        except Exception as e:
            _log.error(f"Error preloading WAN model: {e}")
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
                            "job_id": job.id
                        }

                        # If latents are available in the callback, process and send them
                        if "latents" in callback_kwargs:
                            try:
                                latents = callback_kwargs["latents"]
                                # Process latents using our new processor
                                preview_image = self.latents_processor.process_latents(pipeline_instance, latents)
                                latent_img = self.latents_processor.encode_image(preview_image)
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
                if not pipeline_instance.is_loaded:
                    pipeline_instance.load()
                
                # Generate the video/animation
                result = await asyncio.to_thread(
                    pipeline_instance.predict,
                    job.request,
                    callback_func_base,
                )
                
                # For WAN model, we might get back a video path rather than an image
                if isinstance(result, str) and os.path.exists(result):
                    # Store the video path for later retrieval
                    job.result = result
                    job.state = "completed"
                    
                    # Get frame count and fps from request
                    fps = getattr(job.request, 'fps', 15)
                    num_frames = getattr(job.request, 'num_frames', 81)
                    duration = num_frames / fps if fps > 0 else 0
                    
                    # Notify clients of video ready, matching the old format
                    video_info = {
                        "status": "video_ready",
                        "video_path": result,
                        "fps": fps,
                        "num_frames": num_frames,
                        "duration": duration,
                        "job_id": job.id
                    }
                    await job.notification_queue.put(video_info)
                    
                    # Then send the completion message
                    completion_msg = {
                        "status": "completed", 
                        "image": "",  # Empty image for video jobs
                        "job_id": job.id
                    }
                else:
                    # Convert the PIL image to base64 for sending over JSON
                    buffered = io.BytesIO()
                    result.save(buffered, format="PNG")
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

                # Check if a video was generated despite the error, similar to original implementation
                try:
                    video_path = "/tmp/temp_output.mp4"
                    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                        # Get frame count and fps from request
                        fps = getattr(job.request, 'fps', 15)
                        num_frames = getattr(job.request, 'num_frames', 81)
                        
                        # Send video ready notification with error flag
                        video_info = {
                            "status": "video_ready",
                            "video_path": video_path,
                            "fps": fps,
                            "num_frames": num_frames,
                            "error": "Preview failed but video was generated",
                            "job_id": job.id
                        }
                        await job.notification_queue.put(video_info)
                        _log.info(f"Video ready despite error: {video_path}")
                        
                        # Create a placeholder image for preview
                        placeholder = Image.new('RGB', (480, 480), color=(100, 150, 200))
                        draw = ImageDraw.Draw(placeholder)
                        draw.text((20, 20), "Video generation completed", fill=(255, 255, 255))
                        draw.text((20, 50), "But preview creation failed", fill=(255, 255, 255))
                        draw.text((20, 80), f"Error: {str(e)[:50]}", fill=(255, 255, 255))
                        
                        # Save placeholder as image preview
                        img_bytes = io.BytesIO()
                        placeholder.save(img_bytes, format="PNG")
                        img_bytes.seek(0)
                        encoded_image = base64.b64encode(img_bytes.read()).decode()
                        
                        # Set as result and mark job as completed with warning
                        job.result = video_path
                        job.state = "completed"
                        await job.notification_queue.put({
                            "status": "completed",
                            "image": encoded_image,
                            "processing_time": time.time() - time.time(),  # Not tracking time in this implementation
                            "warning": f"Preview failed but video was generated: {str(e)}",
                            "job_id": job.id
                        })
                        _log.info(f"Worker completed job {job.id} with preview error")
                        job_queue.task_done()
                        continue
                except Exception as inner_e:
                    _log.error(f"Error handling video fallback: {inner_e}")

                # Update job state and notify clients of error
                job.state = "error"
                error_msg = {"status": "error", "message": str(e), "job_id": job.id}
                await job.notification_queue.put(error_msg)

            finally:
                # Mark the task as done
                job_queue.task_done()
    
    def get_capabilities(self) -> ModelCapabilities:
        """Return the capabilities of the WAN model."""
        return ModelCapabilities(
            model_name="WAN",
            supported_parameters={
                "prompt": "string",
                "height": "integer",
                "width": "integer", 
                "negative_prompt": "string",
                "guidance_scale": "float",
                "num_inference_steps": "integer",
                "seed": "integer",
                "num_frames": "integer",
                "fps": "integer"
            },
            parameter_ranges={
                "height": {"min": 320, "max": 768, "step": 64},
                "width": {"min": 576, "max": 1280, "step": 64},
                "guidance_scale": {"min": 1.0, "max": 15.0, "step": 0.1},
                "num_inference_steps": {"min": 20, "max": 50, "step": 1},
                "num_frames": {"min": 16, "max": 100, "step": 1},
                "fps": {"min": 8, "max": 30, "step": 1}
            },
            default_values={
                "height": 480,
                "width": 832,
                "guidance_scale": 5.0,
                "num_inference_steps": 50,
                "num_frames": 80,
                "fps": 15
            }
        )

    async def get_video(self, job_id: str):
        """
        GET endpoint to serve the generated video file for a specific job.
        Returns the video file as a streaming response.
        """
        from common.app_base import jobs
        
        # First check if the job has a result path
        if job_id in jobs and jobs[job_id].result and os.path.exists(jobs[job_id].result):
            video_path = jobs[job_id].result
        else:
            # Fallback to the fixed path as used in the original implementation
            video_path = "/tmp/temp_output.mp4"
        
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
    wan_app = WanApp(args)
    
    # Add the video endpoint
    wan_app.app.get("/video/{job_id}")(wan_app.get_video)
    
    _log.info(f"Starting WAN service on port {args.port}")
    uvicorn.run(
        wan_app.app,
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
    ) 