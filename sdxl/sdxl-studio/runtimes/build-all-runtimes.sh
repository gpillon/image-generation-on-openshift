#!/bin/bash

# Script to build all runtimes with Podman
# Works on both Windows (Git Bash) and Linux environments

# Default flags 
BUILD_SDXL=true
BUILD_FLUX=true 
BUILD_WAN=true

# Cache directory - empty by default (caching disabled)
CACHE_DIR=""

# Function to display help
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "Build container images for SDXL Studio runtimes with caching support."
  echo ""
  echo "Options:"
  echo "  --help              Display this help message and exit"
  echo "  --skip-sdxl         Skip building the SDXL runtime"
  echo "  --skip-flux         Skip building the Flux runtime"
  echo "  --skip-wan          Skip building the WAN runtime"
  echo "  --only-sdxl         Build only the SDXL runtime"
  echo "  --only-flux         Build only the Flux runtime"
  echo "  --only-wan          Build only the WAN runtime"
  echo "  --cache-dir DIR     Use specified directory for pip package cache"
  echo "                      If not specified, caching will be disabled."
  echo ""
  echo "Examples:"
  echo "  $0                  Build all runtimes without caching"
  echo "  $0 --only-sdxl      Build only the SDXL runtime without caching"
  echo "  $0 --cache-dir /tmp/pip_cache  Use custom cache directory"
  exit 0
}

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      show_help
      ;;
    --skip-sdxl)
      BUILD_SDXL=false
      shift
      ;;
    --skip-flux)
      BUILD_FLUX=false
      shift
      ;;
    --skip-wan)
      BUILD_WAN=false
      shift
      ;;
    --only-sdxl)
      BUILD_FLUX=false
      BUILD_WAN=false
      shift
      ;;
    --only-flux)
      BUILD_SDXL=false
      BUILD_WAN=false
      shift
      ;;
    --only-wan)
      BUILD_SDXL=false
      BUILD_FLUX=false
      shift
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Function to print messages with colors
print_message() {
  GREEN='\033[0;32m'
  BLUE='\033[0;34m'
  RED='\033[0;31m'
  YELLOW='\033[1;33m'
  NC='\033[0m' # No Color
  
  case $1 in
    "info")
      echo -e "${BLUE}[INFO]${NC} $2"
      ;;
    "success")
      echo -e "${GREEN}[SUCCESS]${NC} $2"
      ;;
    "error")
      echo -e "${RED}[ERROR]${NC} $2"
      ;;
    "warning")
      echo -e "${YELLOW}[WARNING]${NC} $2"
      ;;
    *)
      echo -e "$2"
      ;;
  esac
}

# Function to check and setup Podman
setup_podman() {
  # Check if Podman is installed
  if ! command -v podman &> /dev/null; then
    print_message "error" "Podman is not installed. Please install it first."
    return 1
  fi
  
  # Check if we're on Windows
  if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    print_message "info" "Windows environment detected, checking Podman Machine..."
    
    # Check if Podman Machine exists
    if ! podman machine list &> /dev/null; then
      print_message "warning" "Podman Machine not found. Initializing..."
      podman machine init
    fi
    
    # Check if Podman Machine is running
    if ! podman machine info &> /dev/null; then
      print_message "warning" "Podman Machine is not running. Starting..."
      podman machine start
      
      # Wait for machine to start
      print_message "info" "Waiting for Podman Machine to initialize..."
      sleep 10
    fi
    
    # Set IS_WINDOWS flag
    IS_WINDOWS=true
  else
    IS_WINDOWS=false
  fi
  
  # Test Podman connectivity
  if ! podman info &> /dev/null; then
    print_message "error" "Cannot connect to Podman. Please check your Podman installation."
    return 1
  fi
  
  print_message "success" "Podman is setup correctly."
  return 0
}

# Function to setup pip cache
setup_pip_cache() {
  # Only setup cache if cache directory is specified
  if [ -n "$CACHE_DIR" ]; then
    # Create cache directory if it doesn't exist
    if [ ! -d "$CACHE_DIR" ]; then
      print_message "info" "Creating pip cache directory at $CACHE_DIR"
      mkdir -p "$CACHE_DIR"
    else
      print_message "info" "Using existing pip cache directory at $CACHE_DIR"
    fi
  else
    print_message "info" "Caching is disabled (no --cache-dir provided)"
  fi
}

# Function to format Windows paths for Podman
format_windows_path_for_podman() {
  local path="$1"
  
  # Extract drive letter (convert from /f/path to f:/path)
  if [[ "$path" =~ ^/([a-zA-Z])/(.*)$ ]]; then
    local drive="${BASH_REMATCH[1]}"
    local rest="${BASH_REMATCH[2]}"
    echo "${drive}:/${rest}"
  else
    # If not a typical Git Bash path, return as is
    echo "$path"
  fi
}

# Function to build a runtime
build_runtime() {
  local runtime_name=$1
  local dockerfile=$2
  local tag=$3
  
  print_message "info" "Building ${runtime_name} runtime..."
  
  # Check if the Containerfile exists
  if [ ! -f "${dockerfile}" ]; then
    print_message "error" "Containerfile not found at ${dockerfile}"
    return 1
  fi
  
  # Create a temporary Containerfile with ENV for pip cache if needed
  local temp_dockerfile="${dockerfile}.temp"
  
  if [ -n "$CACHE_DIR" ]; then
    # With caching
    awk -v cache="/pip_cache" -v token="$HF_TOKEN" '{
      if ($0 ~ /^FROM/) {
        print $0;
        print "ENV PIP_CACHE_DIR=\"" cache "\"";
        print "ENV HF_TOKEN=\"" token "\"";
      } else if ($0 ~ /pip install --prefer-binary/) {
        # Keep the line as is (with caching)
        print $0;
      } else {
        print $0;
      }
    }' "${dockerfile}" > "${temp_dockerfile}"
  else
    # Without caching
    awk -v token="$HF_TOKEN" '{
      if ($0 ~ /^FROM/) {
        print $0;
        print "ENV HF_TOKEN=\"" token "\"";
      } else if ($0 ~ /pip install --prefer-binary/) {
        # Add --no-cache-dir flag
        gsub(/pip install --prefer-binary/, "pip install --no-cache-dir --prefer-binary");
        print $0;
      } else {
        print $0;
      }
    }' "${dockerfile}" > "${temp_dockerfile}"
  fi
  
  # Prepare the volume argument for cache dir
  local cache_mount=""
  if [ -n "$CACHE_DIR" ]; then
    if [ "$IS_WINDOWS" = true ]; then
      # On Windows/Git Bash, format path for Podman
      local podman_path=$(format_windows_path_for_podman "$CACHE_DIR")
      cache_mount="--volume ${podman_path}:/pip_cache:Z"
      print_message "info" "Using cache mount: ${cache_mount}"
    else
      # On Linux, use path as is
      cache_mount="--volume ${CACHE_DIR}:/pip_cache:Z"
    fi
  fi
  
  # Build command with appropriate options
  local build_cmd="podman build --network=host --jobs 4 --retry 3 --retry-delay 10"
  
  if [ -n "$cache_mount" ]; then
    build_cmd="$build_cmd $cache_mount"
  fi
  
  build_cmd="$build_cmd -t ${tag} -f ${temp_dockerfile} ."
  
  print_message "info" "Running build command: $build_cmd"
  
  # Execute the build command
  if eval $build_cmd; then
    print_message "success" "${runtime_name} runtime built successfully"
    # Clean up temporary file
    rm -f "${temp_dockerfile}"
  else
    print_message "error" "Failed to build ${runtime_name} runtime"
    # Clean up temporary file
    rm -f "${temp_dockerfile}"
    return 1
  fi
}

# Ensure we're in the runtimes directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_message "info" "Starting build process for runtimes..."

# Setup Podman
if ! setup_podman; then
  print_message "error" "Failed to setup Podman environment."
  exit 1
fi

# Setup pip cache if needed
setup_pip_cache

# Build SDXL runtime
if $BUILD_SDXL; then
  print_message "info" "Building SDXL runtime..."
  build_runtime "SDXL" "sdxl/Containerfile" "sdxl-runtime" || exit 1
fi

# Build Flux runtime
if $BUILD_FLUX; then
  print_message "info" "Building Flux runtime..."
  build_runtime "Flux" "flux/Containerfile" "flux-runtime" || exit 1
fi

# Build WAN runtime
if $BUILD_WAN; then
  print_message "info" "Building WAN runtime..."
  build_runtime "WAN" "wan/Containerfile" "wan-runtime" || exit 1
fi

print_message "success" "All selected runtimes built successfully!"

# Function to run containers
run_containers() {
  print_message "info" "Do you want to run the containers? (y/n)"
  read -r answer
  
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    if $BUILD_SDXL; then
      print_message "info" "Running SDXL runtime..."
      podman run -it --rm -p 8080:8080 --device nvidia.com/gpu=all --network=host -v "f:\ai\models:/mnt/models:Z" -v "f:\ai\models/tmp_hf:/opt/app-root/src/.cache/:rw"  sdxl-runtime --hf-token=${HF_TOKEN}
    fi
    
    if $BUILD_FLUX; then
      print_message "info" "Running Flux runtime..."
      podman run -it --rm -p 8081:8080 flux-runtime
    fi
    
    if $BUILD_WAN; then
      print_message "info" "Running WAN runtime..."
      podman run -it --rm -p 8082:8080 wan-runtime
    fi
  fi
}

# Uncomment this line if you want to give the option to run containers after building
# run_containers

print_message "info" "Script completed" 