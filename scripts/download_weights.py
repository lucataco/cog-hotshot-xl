#!/usr/bin/env python3
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "hotshotco/SDXL-512",
)

pipe.save_pretrained("/Hotshot-XL/SDXL-512", safe_serialization=True)