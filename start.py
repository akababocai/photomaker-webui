!git clone https://github.com/TencentARC/PhotoMaker.git
%cd PhotoMaker
#Install requirements
!pip install -r requirements.txt
#Install PhotoMaker
!pip install git+https://github.com/TencentARC/PhotoMaker.git

from photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
import torch
import os
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from PIL import Image
import gradio as gr

photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker",filename="photomaker-v1.bin",
repo_type="model")
base_model_path = 'SG161222/RealVisXL_V3.0'
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(base_model_path,torch_dtype=torch.bfloat16,
use_safetensors=True,variant="fp16").to(device)
pipe.load_photomaker_adapter(os.path.dirname(photomaker_path),subfolder="",
weight_name=os.path.basename(photomaker_path),trigger_word="img")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

def generate_image(files,prompt,negative_prompt):
  files_list = [Image.open(f.name) for f in files]
  generator = torch.Generator(device=device).manual_seed(42)
  images = pipe(prompt=prompt,input_id_images=files_list,
      negative_prompt=negative_prompt,num_images_per_prompt=1,
      num_inference_steps=50,start_merge_step=10,generator=generator).images[0]
  return [images]

with gr.Blocks() as demo:
    with gr.Row():
        files = gr.Files(label="Select photos of face",file_types=["image"])
        gallery = gr.Gallery(label="Generated Images")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt",info="Try something like 'a photo of a man/woman img','img'
        is the trigger word.")
        negative_prompt = gr.Textbox(label="Gegative Prompt",
                value="nsfw,lowers,bad anatomy,bad hands,text,error,missing fingers,extra digit,
                fewer digits,cropped, worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,
                username,blurry",)
        submit = gr.Button("Submit")
    submit.click(generate_image,[files,prompt,negative_prompt],[gallery])

demo.launch()


        
    
  

