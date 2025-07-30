from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
import torch
class OODGenerator:
  def __init__(self,
            sd_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
            ip_adapter="h94/IP-Adapter",
            adapter_weight_name="ip-adapter_sd15_light.bin",
            cache_dir="./pretrained_models",
            vae_model = "stabilityai/sd-vae-ft-mse",
            device="cuda"):
    vae = AutoencoderKL.from_pretrained(vae_model,cache_dir=cache_dir,torch_dtype=torch.float16,local_files_only=True,device=device)
    self.device = device
    self.sd_model = sd_model
    self.sdpipe = StableDiffusionPipeline.from_pretrained(
                      sd_model,
                      vae=vae,
                      cache_dir=cache_dir,
                      torch_dtype=torch.float16,
                      safety_checker=None,
                      local_files_only=True
                     ).to(device)
    self.sdpipe._progress_bar_config={"disable": True}
    self.sdpipe.load_ip_adapter(
      ip_adapter,
      device=self.device,
      subfolder="models",
      weight_name=adapter_weight_name,
      local_files_only=True,
      cache_dir=cache_dir
    )

  def generate_images_with_name(self,ip_adapter_image_embeds,class_name,width=256,height=256,seed=42,ip_adapter_scale=0.1,num_inference_steps=50):

    crop_size = 256/8
    images = self.sdpipe(
      prompt=f"",
      negative_prompt=f"{class_name}",
      num_images_per_prompt=1,
      ip_adapter_image_embeds=ip_adapter_image_embeds,
      num_inference_steps=num_inference_steps,
      guidance_scale=12,
      ip_adapter_scale=ip_adapter_scale,
      height = 256 + crop_size*2,
      width = 256 + crop_size*2,
      do_classifier_free_guidance=True
    ).images

    images=  [image.crop((crop_size, crop_size, image.width - crop_size, image.height - crop_size)).resize((width,height)) for image in images]
    return images




  




