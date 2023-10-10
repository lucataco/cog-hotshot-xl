#!/usr/bin/env python3
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "hotshotco/SDXL-512",
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.save_pretrained("./hotshot-xl", safe_serialization=True)
