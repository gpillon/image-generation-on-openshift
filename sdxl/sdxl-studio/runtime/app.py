import asyncio
import base64
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
import os
from PIL import Image

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from classes import GenerationRequest, GenerationResponse, HealthCheckResponse, Job
from diffusers_model import DiffusersPipeline
from flux_model import FluxModelPipeline
from wan_model import WanModelPipeline
from helpers import logging_config, parse_args
from latents_preview import process_latents, process_flux_latents, process_wan_latents
from watermark import add_watermark

# Load local env vars if present
load_dotenv()

# Set up logging
logging_config()
_log = logging.getLogger(__name__)


# Global job dictionary, queue and websocket connections
jobs = {}  # job_id -> Job
job_queue = asyncio.Queue()
queue_list = []  # Maintain an ordered list of job IDs for queue tracking
websocket_connections = {}  # job_id -> set of WebSockets


##################################
# App creation and configuration #
##################################


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model and start background queue processing"""

    # Start the background queue processor
    queue_task = asyncio.create_task(process_queue())

    yield

    # Cancel the background queue processor on shutdown
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        _log.info("Queue processor cancelled.")


args = parse_args()
generation_workers = args.generation_workers
app = FastAPI(title="SDXL Serving runtime", lifespan=lifespan)

# Cors middleware
origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)

#############################
# API Endpoints definitions #
#############################


@app.get("/health")
def health() -> HealthCheckResponse:
    """Health check endpoint."""
    return HealthCheckResponse()


@app.post("/generate")
async def generate(request: GenerationRequest) -> GenerationResponse:
    """
    Instead of immediately processing the generation request,
    create a job and place it on the queue. Return the job id.
    """
    global jobs, job_queue, queue_list

    # Create a unique job id
    job_id = str(uuid.uuid4())
    job = Job(job_id, request)
    jobs[job_id] = job

    # Enqueue the job for processing
    await job_queue.put(job)
    queue_list.append(job_id)

    _log.info(f"Enqueued job {job_id}")

    # Notify all connected clients about queue changes
    await notify_all_queue_positions()

    response = GenerationResponse(job_id=job_id)

    return response

#TODO add a function to cleanup completed jobs that are not being retrieved for more than 1 hour


async def notify_all_queue_positions():
    """Notify all connected WebSocket clients about their queue position."""
    global websocket_connections, queue_list, jobs

    for job_id, connections in websocket_connections.items():
        # Skip jobs that are already being processed or are completed
        if jobs[job_id].state == "processing" or jobs[job_id].state == "completed":
            continue

        position = get_queue_position(job_id)
        message = {"status": "queued", "position": position}
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception as e:
                _log.error(f"Error sending message to WebSocket: {e}")


def get_queue_position(job_id: str) -> int:
    """Return the queue position (1-based) of a job, or -1 if not in queue."""
    return queue_list.index(job_id) + 1 if job_id in queue_list else -1


@app.websocket("/progress/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for clients to subscribe to updates for a given job.
    The server will send JSON messages with progress updates and,
    when complete, the generated image (base64 encoded).
    """
    await websocket.accept()

    if job_id not in jobs:
        await websocket.send_json({"status": "error", "message": "Job not found."})
        await websocket.close()
        return

    job = jobs[job_id]

    # Track active WebSocket connections for this job
    if job_id not in websocket_connections:
        websocket_connections[job_id] = set()
    websocket_connections[job_id].add(websocket)

    try:
        # Send initial queue position
        position = get_queue_position(job_id)
        if position > 0:
            await websocket.send_json({"status": "queued", "position": position})

        # If the job is completed, send the result immediately.
        if job.state == "completed":
            await websocket.send_json({"status": "completed", "image": job.result})
            await websocket.close()
            return

        # Otherwise, listen for notifications.
        while True:
            msg = await job.notification_queue.get()
            await websocket.send_json(msg)
            if msg.get("status") in ("completed", "error"):
                break

    except WebSocketDisconnect:
        _log.info(f"WebSocket disconnected for job {job_id}")

    finally:
        websocket_connections[job_id].remove(websocket)
        if not websocket_connections[job_id]:
            del websocket_connections[job_id]
        # Remove the job from the queue, and delete the job if it's completed.
        if job.state in ("completed", "error"):
            del jobs[job_id]


@app.get("/progress/{job_id}")
async def get_job_status(job_id: str):
    """
    GET endpoint for clients to poll for updates on a given job.
    Returns JSON messages with progress updates and, when completed, the generated image (base64 encoded).
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Get queue position
    position = get_queue_position(job_id)
    if position > 0:
        return {"status": "queued", "position": position}

    # If the job is already completed, return the result immediately.
    if job.state == "completed":
        result = {"status": "completed", "image": job.result}
        del jobs[job_id]
        return result

    # Otherwise, return the latest available status
    # Empty the notification_queue and keep only the last message
    msg = None
    while not job.notification_queue.empty():
        msg = await job.notification_queue.get()
    return msg


@app.get("/video/{job_id}")
async def get_video(job_id: str):
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


##################################
# Background queue processing    #
##################################


async def worker(worker_id, job_queue, pipeline_instance):
    """
    Worker function that processes jobs from the queue.
    Callback functions are used by predict to notify the client of progress.
    """
    while True:
        job = await job_queue.get()
        queue_list.remove(job.id)

        try:
            job.state = "processing"
            _log.info(f"Worker {worker_id} processing job {job.id}")

            # Notify clients about queue updates
            await notify_all_queue_positions()

            await job.notification_queue.put(
                {"status": "processing", "message": "Job is processing."}
            )
            # Get the current event loop
            loop = asyncio.get_event_loop()

            # Define a callback function to send progress updates to the client.
            def callback_func_base(_pipe, step, _timestep, callback_kwargs):
                latents = callback_kwargs["latents"]
                # Use appropriate latent processing function based on pipeline type
                if isinstance(pipeline_instance, FluxModelPipeline):
                    base64_image = process_flux_latents(pipeline_instance, latents)
                elif isinstance(pipeline_instance, WanModelPipeline):
                    base64_image = process_wan_latents(pipeline_instance, latents)
                else:
                    base64_image = process_latents(pipeline_instance, latents)
                
                # Calculate progress percentage for more accurate reporting
                total_steps = job.request.num_inference_steps
                progress_pct = int((step + 1) / total_steps * 100)
                
                future = asyncio.run_coroutine_threadsafe(
                    job.notification_queue.put(
                        {
                            "pipeline": "base",
                            "status": "progress",
                            "step": step,
                            "progress": progress_pct,
                            "image": base64_image,
                        }
                    ),
                    loop,
                )
                future.result()  # Wait for the coroutine to finish
                return {}

            def callback_func_refiner(_pipe, step, _timestep, callback_kwargs):
                latents = callback_kwargs["latents"]
                # Use appropriate latent processing function based on pipeline type
                if isinstance(pipeline_instance, FluxModelPipeline):
                    base64_image = process_flux_latents(pipeline_instance, latents)
                elif isinstance(pipeline_instance, WanModelPipeline):
                    base64_image = process_wan_latents(pipeline_instance, latents)
                else:
                    base64_image = process_latents(pipeline_instance, latents)
                future = asyncio.run_coroutine_threadsafe(
                    job.notification_queue.put(
                        {
                            "pipeline": "refiner",
                            "status": "progress",
                            "step": step,
                            "image": base64_image,
                        }
                    ),
                    loop,
                )
                future.result()  # Wait for the coroutine to finish
                return {}

            # Run the prediction in a thread to avoid blocking the event loop.
            start_time = time.time()
            image = await asyncio.to_thread(
                pipeline_instance.predict,
                job.request,
                callback_func_base,
                callback_func_refiner,
            )
            processing_time = time.time() - start_time

            # Prepare image bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            encoded_image = base64.b64encode(img_bytes.read()).decode("utf-8")

            # Add watermark to the base64 encoded image if it's enabled 
            
            enable_watermark = os.getenv("ENABLE_WATERMARK", "true")
            if enable_watermark == "true":
                watermark_text = os.getenv("WATERMARK_TEXT", "AI-generated Image. Demo purposes only. More info at red.ht/maas")
                watermarked_image = add_watermark(encoded_image, watermark_text)
                job.result = watermarked_image
            else:
                job.result = encoded_image

            # For WAN models, send additional video info
            if isinstance(pipeline_instance, WanModelPipeline):
                # Get the video path
                video_path = os.path.abspath("/tmp/temp_output.mp4")
                if os.path.exists(video_path):
                    video_info = {
                        "status": "video_ready",
                        "video_path": video_path,
                        "fps": getattr(job.request, 'fps', 15),
                        "num_frames": getattr(job.request, 'num_frames', 81),
                        "duration": getattr(job.request, 'num_frames', 81) / getattr(job.request, 'fps', 15),
                    }
                    await job.notification_queue.put(video_info)
                    _log.info(f"Video ready: {video_path}, duration: {video_info['duration']:.2f}s")
                else:
                    _log.warning(f"Video file not found at {video_path}")

            # Handle the result and notify the client
            await job.notification_queue.put(
                {
                    "status": "completed",
                    "image": job.result,
                    "processing_time": processing_time,
                }
            )
            job.state = "completed"
            _log.info(
                f"Worker {worker_id} completed job {job.id} in {processing_time:.2f} seconds"
            )

        except Exception as e:
            _log.error(f"Worker {worker_id} failed to process job {job.id}: {e}")
            job.state = "failed"
            
            # Check if this was a video generation job and if the video file exists
            # despite the error (which might be just in preview image creation)
            try:
                if isinstance(pipeline_instance, WanModelPipeline):
                    video_path = os.path.abspath("/tmp/temp_output.mp4")
                    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                        video_info = {
                            "status": "video_ready",
                            "video_path": video_path,
                            "fps": getattr(job.request, 'fps', 15),
                            "num_frames": getattr(job.request, 'num_frames', 81),
                            "error": "Preview failed but video was generated",
                        }
                        await job.notification_queue.put(video_info)
                        _log.info(f"Video ready despite error: {video_path}")
                        
                        # Create a placeholder image for preview
                        placeholder = Image.new('RGB', (480, 480), color=(100, 150, 200))
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(placeholder)
                        draw.text((20, 20), "Video generation completed", fill=(255, 255, 255))
                        draw.text((20, 50), "But preview creation failed", fill=(255, 255, 255))
                        draw.text((20, 80), f"Error: {str(e)[:50]}", fill=(255, 255, 255))
                        
                        # Save placeholder as image preview
                        img_bytes = io.BytesIO()
                        placeholder.save(img_bytes, format="PNG")
                        img_bytes.seek(0)
                        encoded_image = base64.b64encode(img_bytes.read()).decode("utf-8")
                        
                        # Set as result and mark job as completed with warning
                        job.result = encoded_image
                        job.state = "completed"
                        await job.notification_queue.put({
                            "status": "completed",
                            "image": job.result,
                            "processing_time": time.time() - start_time,
                            "warning": f"Preview failed but video was generated: {str(e)}"
                        })
                        _log.info(f"Worker {worker_id} completed job {job.id} with preview error")
                        job_queue.task_done()
                        continue
            except Exception as inner_e:
                _log.error(f"Error handling video fallback: {inner_e}")
            
            # If we got here, send the error message
            await job.notification_queue.put({"status": "failed", "message": str(e)})

        finally:
            job_queue.task_done()


async def process_queue():
    """
    Background task that continuously processes jobs from the queue.
    """
    global job_queue, args

    _log.info(f"Device: {args.device}")
    _log.info(f"Model Type: {args.model_type}")
    _log.info(f"Creating {generation_workers} worker(s)...")
    _log.info(f"Model path: {args.model_id}, Single file model: {args.single_file_model}")

    # Create a pool of workers
    workers = []
    for i in range(generation_workers):
        _log.info(f"Initializing worker {i}...")
        try:
            # Select the appropriate pipeline based on model_type
            if args.model_type == "flux":
                _log.info(f"Worker {i}: Creating Flux pipeline...")
                pipeline_instance = FluxModelPipeline(args)
            elif args.model_type == "wan":
                _log.info(f"Worker {i}: Creating WAN pipeline...")
                pipeline_instance = WanModelPipeline(args)
            else:
                _log.info(f"Worker {i}: Creating SDXL pipeline...")
                pipeline_instance = DiffusersPipeline(args)
                
            _log.info(f"Worker {i}: Loading model...")
            pipeline_instance.load()
            _log.info(f"Worker {i}: Model loaded successfully!")
            worker_task = asyncio.create_task(worker(i, job_queue, pipeline_instance))
            workers.append(worker_task)
            _log.info(f"Worker {i} initialized and started")
        except Exception as e:
            _log.error(f"Error initializing worker {i}: {str(e)}")
            import traceback
            _log.error(traceback.format_exc())

    if not workers:
        _log.error("No workers were initialized successfully! Jobs will remain queued.")
    
    # Wait for all workers to complete (they won't, as they run indefinitely)
    await asyncio.gather(*workers)


# Launch the FastAPI server
if __name__ == "__main__":
    from uvicorn import run

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    run(
        "app:app",
        host="0.0.0.0",
        port=args.port,
        timeout_keep_alive=600,
        reload=args.reload,
    )