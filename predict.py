# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


from cog import BasePredictor, Input, Path
import os
import sys
sys.path.extend(['/Hotshot-XL'])

import torch
import tempfile
from hotshot_xl.pipelines.hotshot_xl_pipeline import HotshotXLPipeline
from hotshot_xl.pipelines.hotshot_xl_controlnet_pipeline import HotshotXLControlNetPipeline
from hotshot_xl.models.unet import UNet3DConditionModel
import torchvision.transforms as transforms
from einops import rearrange
from hotshot_xl.utils import save_as_gif, extract_gif_frames_from_midpoint, scale_aspect_fill
from torch import autocast
from diffusers import ControlNetModel
from contextlib import contextmanager
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

SCHEDULERS = {
    'EulerAncestralDiscreteScheduler': EulerAncestralDiscreteScheduler,
    'EulerDiscreteScheduler': EulerDiscreteScheduler,
}

MODEL_NAME = "hotshotco/Hotshot-XL"
MODEL_CACHE = "model-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pipe_line_args = {
            "torch_dtype": torch.float16,
            "use_safetensors": True
        }
        self.pipe = HotshotXLPipeline.from_pretrained(
            MODEL_NAME, 
            **pipe_line_args,
            cache_dir=MODEL_CACHE
        ).to('cuda')

    def to_pil_images(self, video_frames: torch.Tensor, output_type='pil'):
        to_pil = transforms.ToPILImage()
        video_frames = rearrange(video_frames, "b c f w h -> b f c w h")
        bsz = video_frames.shape[0]
        images = []
        for i in range(bsz):
            video = video_frames[i]
            for j in range(video.shape[0]):
                if output_type == "pil":
                    images.append(to_pil(video[j]))
                else:
                    images.append(video[j])
        return images

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="a camel smoking a cigarette, hd, high quality"),
        negative_prompt: str = Input(description="Negative prompt", default="blurry"),
        scheduler: str = Input(
            default="EulerAncestralDiscreteScheduler",
            choices=[
                "EulerAncestralDiscreteScheduler",
                "EulerDiscreteScheduler",
            ],
            description="Select a Scheduler",
        ),
        steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        device = torch.device("cuda")

        # Default text2Gif parameters
        # seed = 455
        # steps = 30
        width = 672
        height = 384
        target_width = 512
        target_height = 512
        og_width = 1920
        og_height = 1080
        video_length = 8
        video_duration = 1000
        # scheduler = "EulerAncestralDiscreteScheduler"

        device = torch.device("cuda")
        pipe_line_args = {
            "torch_dtype": torch.float16,
            "use_safetensors": True
        }
        PipelineClass = HotshotXLPipeline

        pipe = PipelineClass.from_pretrained(
            MODEL_NAME, 
            **pipe_line_args,
            cache_dir=MODEL_CACHE
        ).to(device)
        
        SchedulerClass = SCHEDULERS[scheduler]
        if SchedulerClass is not None:
            pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config)

        pipe.enable_xformers_memory_efficient_attention()
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator().manual_seed(seed)

        kwargs = {}
        images = pipe(prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            original_size=(og_width, og_height),
            target_size=(target_width, target_height),
            num_inference_steps=steps,
            video_length=video_length,
            generator=generator,
            output_type="tensor", **kwargs
        ).videos

        images = self.to_pil_images(images, output_type="pil")

        output = "output.gif"
        save_as_gif(images, output, duration=video_duration // video_length)

        out_path = Path(tempfile.mkdtemp()) / "out.mp4"
        os.system("ffmpeg -i output.gif -movflags faststart -pix_fmt yuv420p -qp 17 "+ str(out_path))

        return Path(out_path)
