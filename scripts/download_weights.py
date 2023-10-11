#!/usr/bin/env python3
from hotshot_xl.pipelines.hotshot_xl_pipeline import HotshotXLPipeline
import torch

pipe = HotshotXLPipeline.from_pretrained(
    "hotshotco/Hotshot-XL",
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.save_pretrained("./hotshot-xl", safe_serialization=True)
