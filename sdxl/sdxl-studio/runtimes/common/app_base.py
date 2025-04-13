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

from .classes import GenerationRequest, GenerationResponse, HealthCheckResponse, Job, ModelCapabilities

# Load local env vars if present
load_dotenv()

# Set up logging
_log = logging.getLogger(__name__)

# Global job dictionary, queue and websocket connections
jobs = {}  # job_id -> Job
job_queue = asyncio.Queue()
queue_list = []  # Maintain an ordered list of job IDs for queue tracking
websocket_connections = {}  # job_id -> set of WebSockets


class BaseApp:
    """Base class for all model runtimes."""
    
    def __init__(self, model_name, lifespan_func=None, model_instance=None):
        self.model_name = model_name
        self.generation_workers = 1
        self.app = None
        self.model_instance = model_instance
        self.setup_app(lifespan_func)
    
    def setup_app(self, lifespan_func=None):
        """Set up FastAPI application with common endpoints."""
        
        if lifespan_func:
            self.app = FastAPI(title=f"{self.model_name} Serving Runtime", lifespan=lifespan_func)
        else:
            self.app = FastAPI(title=f"{self.model_name} Serving Runtime")
        
        # Cors middleware
        origins = ["*"]
        methods = ["*"]
        headers = ["*"]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=methods,
            allow_headers=headers,
        )
        
        # Add common endpoints
        self.app.get("/health")(self.health)
        self.app.post("/generate")(self.generate)
        self.app.get("/progress/{job_id}")(self.get_job_status)
        self.app.websocket("/progress/{job_id}")(self.websocket_endpoint)
        self.app.get("/capabilities")(self.get_capabilities)
    
    def health(self) -> HealthCheckResponse:
        """Health check endpoint."""
        return HealthCheckResponse()
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
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
        await self.notify_all_queue_positions()

        response = GenerationResponse(job_id=job_id)
        return response
    
    async def get_job_status(self, job_id: str):
        """
        GET endpoint for clients to poll for updates on a given job.
        Returns JSON messages with progress updates and, when completed, the generated image (base64 encoded).
        """
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]

        # Get queue position
        position = self.get_queue_position(job_id)
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
            try:
                msg = await job.notification_queue.get()
            except Exception as e:
                _log.error(f"Error getting notification: {e}")
        return msg
    
    async def websocket_endpoint(self, websocket: WebSocket, job_id: str):
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
            position = self.get_queue_position(job_id)
            if position > 0:
                await websocket.send_json({"status": "queued", "position": position})

            # If the job is completed, send the result immediately.
            if job.state == "completed":
                await websocket.send_json({"status": "completed", "image": job.result})
                await websocket.close()
                return

            # Otherwise, listen for notifications.
            while True:
                try:
                    msg = await job.notification_queue.get()
                    
                    # Check if any field in the message is a PIL Image and convert it to base64
                    for key in msg:
                        if isinstance(msg[key], Image.Image):
                            img_bytes = io.BytesIO()
                            msg[key].save(img_bytes, format="PNG")
                            img_bytes.seek(0)
                            msg[key] = base64.b64encode(img_bytes.read()).decode("utf-8")
                    
                    await websocket.send_json(msg)
                    if msg.get("status") in ("completed", "error"):
                        break
                except Exception as e:
                    _log.error(f"Error in websocket communication: {e}")
                    break

        except WebSocketDisconnect:
            _log.info(f"WebSocket disconnected for job {job_id}")

        finally:
            websocket_connections[job_id].remove(websocket)
            if not websocket_connections[job_id]:
                del websocket_connections[job_id]
            # Remove the job from the queue, and delete the job if it's completed.
            if job.state in ("completed", "error"):
                try:
                    del jobs[job_id]
                except KeyError:
                    pass
    
    def get_queue_position(self, job_id: str) -> int:
        """Return the queue position (1-based) of a job, or -1 if not in queue."""
        return queue_list.index(job_id) + 1 if job_id in queue_list else -1
    
    async def notify_all_queue_positions(self):
        """Notify all connected WebSocket clients about their queue position."""
        global websocket_connections, queue_list, jobs

        for job_id, connections in websocket_connections.items():
            # Skip jobs that are already being processed or are completed
            if job_id not in jobs or jobs[job_id].state == "processing" or jobs[job_id].state == "completed":
                continue

            position = self.get_queue_position(job_id)
            message = {"status": "queued", "position": position}
            for ws in connections:
                try:
                    await ws.send_json(message)
                except Exception as e:
                    _log.error(f"Error sending message to WebSocket: {e}")
    
    def get_capabilities(self) -> ModelCapabilities:
        """
        Returns the model's capabilities, including supported parameters,
        valid ranges, and default values.
        """
        raise NotImplementedError("Each model must implement its own capabilities endpoint")
    
    async def process_queue(self):
        """
        Process the job queue in the background.
        This should be implemented by each specific runtime.
        """
        raise NotImplementedError("Each model must implement its own queue processing logic") 