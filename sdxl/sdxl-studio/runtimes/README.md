# SDXL Studio Runtimes

This directory contains the runtime implementations for various AI image and video generation models:

## Structure

- **common/**: Shared code and utilities used by all runtimes
  - `app_base.py`: Base FastAPI server implementation
  - `classes.py`: Shared data models
  - `helpers.py`: Utility functions
  - `requirements.txt`: Shared Python dependencies
  
- **sdxl/**: Stable Diffusion XL runtime
  - `app.py`: Main application for SDXL
  - `diffusers_model.py`: SDXL model implementation
  - `Containerfile`: Container build specification

- **flux/**: Flux model runtime
  - `app.py`: Main application for Flux
  - `flux_model.py`: Flux model implementation
  - `Containerfile`: Container build specification

- **wan/**: WAN video model runtime
  - `app.py`: Main application for WAN
  - `wan_model.py`: WAN model implementation
  - `Containerfile`: Container build specification

## Building and Running

### Building Containers

Each runtime has its own Containerfile:

```bash
# Build SDXL runtime
podman build -f sdxl/Containerfile -t sdxl-runtime:latest .

# Build Flux runtime
podman build -f flux/Containerfile -t flux-runtime:latest .

# Build WAN runtime
podman build -f wan/Containerfile -t wan-runtime:latest .
```

### Running Locally

To run the runtimes locally without containers:

```bash
# Run SDXL runtime
cd sdxl/sdxl-studio
python runtimes/sdxl/app.py

# Run Flux runtime
cd sdxl/sdxl-studio
python runtimes/flux/app.py

# Run WAN runtime
cd sdxl/sdxl-studio
python runtimes/wan/app.py --model-id=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

## Configuration

Each runtime supports the following environment variables:

- `GENERATION_WORKERS`: Number of worker processes (default: 1)
- `PORT`: Port number to run the server on (default: 8080)
- `RELOAD`: Enable auto-reload for development (default: False)
- `MODEL_ID`: Model ID or path to load (default: "/mnt/models")
- `SINGLE_FILE_MODEL`: Name of a single file model to load (default: None)
- `DEVICE`: Device to use (default: "cuda")

SDXL-specific options:
- `USE_REFINER`: Whether to use the refiner model (default: False)
- `REFINER_ID`: Refiner model ID to load (default: None)
- `REFINER_SINGLE_FILE_MODEL`: Name of a single file refiner model (default: None)

## API Endpoints

All runtimes provide the following API endpoints:

- `GET /health`: Health check endpoint
- `POST /generate`: Generate an image or video based on a prompt
- `GET /progress/{job_id}`: Get the progress of a generation job
- `WebSocket /progress/{job_id}`: WebSocket endpoint for real-time progress updates
- `GET /capabilities`: Get the capabilities of the model

The WAN model also provides:
- `GET /video/{job_id}`: Get the generated video for a job 