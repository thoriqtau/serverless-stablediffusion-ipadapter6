import torch
import runpod
import base64
import traceback

from io import BytesIO
from PIL import Image

from runpod.serverless.modules.rp_logger import RunPodLogger

from diffusers import (EulerAncestralDiscreteScheduler,
                       StableDiffusionPipeline,
                       StableDiffusionImg2ImgPipeline)
from transformers import CLIPVisionModelWithProjection

from detect import face_detection
from utils import load_image, convert_from_image_to_cv2

# Global variables
IMAGE_ENCODER_PATH = "./CLIP-ViT-H-14-laion2B-s32B-b79K"
BASE_MODEL_PATH = "./CyberRealistic_V9_FP32"
IPADAPTER_FOLDER = "./ipadapter"
IPADAPTER_SUBFOLDER = "./ipadapter_sd15"
IPADATER_PLUS = "./ip-adapter-plus_sd15.bin"
IPADAPTER_FACE = "./ip-adapter-full-face_sd15.bin"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if str(DEVICE).__contains__('cuda') else torch.float32

logger = RunPodLogger()

# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def get_pipeline():
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IMAGE_ENCODER_PATH,
        torch_dtype=DTYPE,
        use_safetensors=True,
        local_files_only=True
    ).to(DEVICE)

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=DTYPE,
        image_encoder=image_encoder,
        use_safetensors=True,
        local_files_only=True
    ).to(DEVICE)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.load_ip_adapter(
        IPADAPTER_FOLDER,
        subfolder=IPADAPTER_SUBFOLDER,
        weight_name=[
            IPADATER_PLUS,
            IPADAPTER_FACE,
        ]
    )
    pipe.set_ip_adapter_scale([1.0,  1.0])

    return pipe

PIPELINE_TEXT2IMG = get_pipeline()

def generate_image(job_id,
                   prompt=None, 
                   negative_prompt=None, 
                   face_image=None,
                   style_image=None,
                   num_inference_steps=80,
                   guidance_scale=1.0,
                   width=512,
                   height=512,
                   ):

    if face_image is None:
        raise Exception("Cannot find input face image! Please upload the face image")
    
    face_image = load_image(face_image)
    style_image = load_image(style_image)

    face_image_cv2 = convert_from_image_to_cv2(face_image)
    style_image_cv2 = convert_from_image_to_cv2(style_image)

    face_crop, style_image = face_detection(face_image_cv2, style_image_cv2)

    if prompt == None:
        prompt = "a portrait of a person, looking directly at the camera, symmetrical face"
    
    if negative_prompt == None:
        negative_prompt = "(deformed, blurry, long neck, bad collar, cgi, bad anatomy, big body)"

    generator = torch.Generator(device=DEVICE).manual_seed(0)

    logger.info('Start inference...', job_id)
    logger.info(f'Prompt: {prompt})', job_id)
    logger.info(f'Negative Prompt: {negative_prompt}', job_id)

    images = PIPELINE_TEXT2IMG(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=[style_image, face_crop],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        num_images_per_prompt=1
    ).images

    upscaled_image = images[0].resize((1024, 1024), Image.LANCZOS)

    return upscaled_image

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get('prompt')
        negative_prompt = job_input.get('negative_prompt')
        face_image = job_input.get('face_image')
        style_image = job_input.get('style_image')
        num_inference_steps = job_input.get('num_inference_steps')
        guidance_scale = job_input.get('guidance_scale')
        height = job_input.get('height')
        width = job_input.get('width')

        images = generate_image(
            job['id'],
            prompt,
            negative_prompt,
            face_image,
            style_image,
            num_inference_steps, 
            guidance_scale,
            height,
            width
        )

        result_image = images
        output_buffer = BytesIO()
        result_image.save(output_buffer, format='PNG')
        image_data = output_buffer.getvalue()

        return {
            'image': base64.b64encode(image_data).decode('utf-8')
        }
    
    except Exception as e:
        logger.error(f'An exception was raised: {e}')

        return {
            'error': str(e),
            'output': traceback.format_exc(),
            'refresh_worker': True
        }
   
# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
