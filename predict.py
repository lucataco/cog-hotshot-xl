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
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
)

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

SCHEDULERS = {
    'DDIMScheduler': DDIMScheduler,
    'DPMSolverMultistepScheduler': DPMSolverMultistepScheduler,
    'HeunDiscreteScheduler': HeunDiscreteScheduler,
    'KarrasDPM': KarrasDPM,
    'EulerAncestralDiscreteScheduler': EulerAncestralDiscreteScheduler,
    'EulerDiscreteScheduler': EulerDiscreteScheduler,
    'PNDMScheduler': PNDMScheduler,
}

HOTSHOTXL_CACHE = "hotshot-xl"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = HotshotXLPipeline.from_pretrained(
            HOTSHOTXL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True
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
        width: int = Input(
            description="Width of the output",
            default=672,
            choices=[
                256, 320, 384, 448, 512, 576, 640, 672, 704, 768, 832, 896, 960, 1024,
            ],
        ),
        height: int = Input(
            description="Height of the output",
            default=384,
            choices=[
                256, 320, 384, 448, 512, 576, 640, 672, 704, 768, 832, 896, 960, 1024,
            ],
        ),
        scheduler: str = Input(
            default="EulerAncestralDiscreteScheduler",
            choices=[
                "DDIMScheduler",
                "DPMSolverMultistepScheduler",
                "HeunDiscreteScheduler",
                "KarrasDPM",
                "EulerAncestralDiscreteScheduler",
                "EulerDiscreteScheduler",
                "PNDMScheduler",
            ],
            description="Select a Scheduler",
        ),
        steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        mp4: bool = Input(
            description="Save as mp4, False for GIF", default=False
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Default text2Gif parameters
        target_width = 512
        target_height = 512
        og_width = 1920
        og_height = 1080
        video_length = 8
        video_duration = 1000
        pipe = self.pipe

        SchedulerClass = SCHEDULERS[scheduler]
        if SchedulerClass is not None:
            pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config)

        # pipe.enable_xformers_memory_efficient_attention()
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

        out_path = "output.gif"
        save_as_gif(images, out_path, duration=video_duration // video_length)

        if mp4:
            out_dir = Path(tempfile.mkdtemp())
            out_path = out_dir / "out.mp4"
            for i, image in enumerate(images):
                image.save(str(out_dir / f"{i:03}.png"))
            os.system(f"ffmpeg -pattern_type glob -i '{str(out_dir)}/*.png' -movflags faststart -pix_fmt yuv420p -qp 17 "+ str(out_path))

        return Path(out_path)
