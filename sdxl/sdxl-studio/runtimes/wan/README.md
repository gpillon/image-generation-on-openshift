# WAN Text-to-Video Model

This directory contains the runtime implementation for WAN (We Are Narrative) text-to-video model.

## Features

- Generate videos from text prompts
- Real-time generation progress with frame previews
- API endpoints for video serving and generation
- Supports different model backends (Hugging Face or local models)

## Usage

### Running locally

```bash
# Install dependencies
pip install -r ../common/requirements.txt
pip install -r requirements.txt

# Run the server
python app.py --model-id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --port 8080
```

### Environment variables

The server can be configured with the following environment variables:

- `MODEL_ID`: Path to the HuggingFace model ID (default: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
- `SINGLE_FILE_MODEL`: Path to a single file model (safetensors file)
- `FPS`: Frames per second for video output (default: 15)
- `NUM_FRAMES`: Default number of frames to generate (default: 80)
- `PORT`: Port to run the server on (default: 8080)
- `DEVICE`: Device to run inference on (default: "cuda", options: "cuda", "cpu")
- `USE_WATERMARK`: Whether to add watermark to generated outputs (default: "False")
- `WATERMARK_TEXT`: Text to use for watermark (default: "SDXL Studio WAN")

### Building with Docker

```bash
# Build the image
docker build -f wan/Dockerfile -t wan-model .

# Run the container
docker run -p 8080:8080 --gpus all wan-model
```

## API Endpoints

- `POST /generate`: Generate a video from a text prompt
- `GET /video/{job_id}`: Get a generated video by job ID
- `GET /capabilities`: Get model capabilities
- `GET /queue/{job_id}`: Get status of a job
- `GET /queue`: Get status of all jobs in the queue
- `POST /queue/{job_id}/cancel`: Cancel a queued job

## Example API Usage

```python
import requests
import json

# Generate a video
response = requests.post(
    "http://localhost:8080/generate",
    json={
        "prompt": "A spaceship flying through an asteroid field",
        "num_inference_steps": 50,
        "num_frames": 60,
        "fps": 15,
        "guidance_scale": 7.5,
        "seed": 42
    }
)

job_id = response.json()["job_id"]

# Get the video URL
response = requests.get(f"http://localhost:8080/queue/{job_id}")
video_url = response.json().get("video_url")

# Download the video
if video_url:
    video_data = requests.get(f"http://localhost:8080{video_url}").content
    with open("output.mp4", "wb") as f:
        f.write(video_data)
``` 