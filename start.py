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
from photomaker import PhotoMakerStablediffusionXLPipeline
from PIL import Image
import gradio as gr
photomaker_path = hf_hub_download(repo_id = "TencentARC/PhotoMaker",filename = "photomaker-v1.bin",
repo_type = "model")
base_model_path = 'SG161222/RealVisXL_V3.0'
device = 
