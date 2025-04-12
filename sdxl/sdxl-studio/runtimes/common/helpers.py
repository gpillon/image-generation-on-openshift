import argparse
import logging
import os


class BaseArgumentParser:
    """Base class for argument parsing with common arguments for all models."""
    
    def __init__(self, description="Base Model API Server"):
        """Initialize with an argument parser and add common arguments."""
        self.parser = argparse.ArgumentParser(description=description)
        self._add_common_arguments()
    
    def _add_common_arguments(self):
        """Add arguments common to all models."""
        # Server and workers configuration
        self.parser.add_argument(
            "--generation-workers",
            type=int,
            default=int(os.getenv("GENERATION_WORKERS", "1")),
            help="Number of image processing workers",
        )
        self.parser.add_argument(
            "--port",
            type=int,
            default=int(os.getenv("PORT", "8080")),
            help="Port number to run the server on",
        )
        self.parser.add_argument(
            "--host", 
            type=str, 
            default="0.0.0.0",
            help="Host to run the server on"
        )
        self.parser.add_argument(
            "--reload",
            type=bool,
            default=bool(os.getenv("RELOAD", "False").lower() in ("true", "1", "t")),
            help="Enable auto-reload",
        )
        
        # Model configuration
        self.parser.add_argument(
            "--model-id",
            type=str,
            default=os.getenv("MODEL_ID", None),
            help="Model ID to load (HuggingFace model ID or path)",
        )
        self.parser.add_argument(
            "--single-file-model",
            type=str,
            default=os.getenv("SINGLE_FILE_MODEL", None),
            help="Name of a single file model to load",
        )
        self.parser.add_argument(
            "--device",
            type=str,
            default=os.getenv("DEVICE", "cuda"),
            choices=["cuda", "cpu"],
            help="Device to run inference on",
        )
        
        # Logging
        self.parser.add_argument(
            "--debug", 
            action="store_true", 
            help="Enable debug mode"
        )
        self.parser.add_argument(
            "--log-level", 
            type=str, 
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level"
        )
    
    def parse_args(self):
        """Parse and return the arguments."""
        args = self.parser.parse_args()
        
        # Set up logging based on args
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {args.log_level}")
        
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        # Set environment variable for PYTHONPATH
        os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        return args


# For backward compatibility - use the BaseArgumentParser directly
def parse_args():
    """Legacy function to parse arguments with default settings."""
    parser = BaseArgumentParser()
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