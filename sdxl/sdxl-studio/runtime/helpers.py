import argparse
import logging
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion XL on FastAPI.")
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=int(os.getenv("GENERATION_WORKERS", "1")),
        help="Number of image processing workers",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8080")),
        help="Port number to run the server on",
    )
    parser.add_argument(
        "--reload",
        type=bool,
        default=bool(os.getenv("RELOAD", "False").lower() in ("true", "1", "t")),
        help="Enable auto-reload",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("MODEL_ID", "/mnt/models"),
        help="Model ID to load (default: /mnt/models, adapt if you use the refiner model)",
    )
    parser.add_argument(
        "--single-file-model",
        type=str,
        default=os.getenv("SINGLE_FILE_MODEL", None),
        help="Name of a single file model to load",
    )
    parser.add_argument(
        "--use-refiner",
        type=bool,
        default=bool(os.getenv("USE_REFINER", "False").lower() in ("true", "1", "t")),
        help="Use the refiner model",
    )
    parser.add_argument(
        "--refiner-id",
        type=str,
        default=os.getenv("REFINER_ID", None),
        help="Refiner model ID to load (or adapt from /mnt/models)",
    )
    parser.add_argument(
        "--refiner-single-file-model",
        type=str,
        default=os.getenv("REFINER_SINGLE_FILE_MODEL", None),
        help="Name of a single file refiner model to load",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "cuda"),
        help="Device to use, including offloading. Valid values are: 'cuda' (default), 'enable_model_cpu_offload', 'enable_sequential_cpu_offload', 'cpu' (works but unusable...)",
    )
    return parser.parse_args()


# Set up custom logging as we'll be intermixes with FastAPI/Uvicorn's logging
class ColoredLogFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.INFO: "\033[92m",  # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[95m",  # Magenta
    }
    RESET_CODE = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self.RESET_CODE}"
        return super().format(record)


def logging_config():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format="%(levelname)s:\t%(asctime)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Override the formatter with the custom ColoredLogFormatter
    root_logger = logging.getLogger()  # Get the root logger
    for handler in root_logger.handlers:  # Iterate through existing handlers
        if handler.formatter:
            handler.setFormatter(ColoredLogFormatter(handler.formatter._fmt))
