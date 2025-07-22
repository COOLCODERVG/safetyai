import os
import time
import uuid
from PIL import Image
import io
import logging
from datetime import datetime
import base64
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager


# Force full precision mode and enable HF transfer
os.environ["COMMANDLINE_ARGS"] = '--precision full --no-half'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#IMPORTANT: UNINSTALL TORCH AND INSTALL TORCH FOR CUDA 11.8 IF U HAVE A GPU

class ThreatDetectionSystem:
   def __init__(self):
       self.model = None
       # Check for Apple Silicon (MPS), CUDA, or fallback to CPU
       if torch.backends.mps.is_available():
           self.device = "mps"
           logger.info("Using Apple Silicon (MPS) device")
           # Conservative settings for MPS
           self.min_process_interval = 1.5
           self.cache_timeout = 2.0
           self.process_every_n_frames = 4
           self.detection_timeout = 3.0
       elif torch.cuda.is_available():
           self.device = "cuda"
           logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
           # Set CUDA device
           torch.cuda.set_device(0)
           # Aggressive settings for CUDA
           self.min_process_interval = 0.1
           self.cache_timeout = 0.5
           self.process_every_n_frames = 1
           self.detection_timeout = 1.0
       else:
           self.device = "cpu"
           logger.info("Using CPU device")
           # Force FP32 for Windows CPU
           torch.set_default_dtype(torch.float32)
           torch.set_default_device('cpu')
           # Conservative settings for CPU
           self.min_process_interval = 1.5
           self.cache_timeout = 2.0
           self.process_every_n_frames = 4
           self.detection_timeout = 3.0
       self.last_process_time = 0
       self.max_image_size = 256
       self.result_cache = {}
       self.frame_counter = 0
       self.last_detections = []
       self.initialize_model()