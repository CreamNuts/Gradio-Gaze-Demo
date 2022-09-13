from unicodedata import mirrored

import cv2
import gradio as gr
import numpy as np
from omegaconf import OmegaConf

from ptgaze.demo import Demo
from ptgaze.utils import generate_dummy_camera_params

config = OmegaConf.load("./config.yaml")
preinit_image = cv2.imread("./img.jpg")
preinit_image = cv2.cvtColor(preinit_image, cv2.COLOR_BGR2RGB)
if config.gaze_estimator.camera_params is None:
    generate_dummy_camera_params(config, preinit_image)
gaze = Demo(config)
# gaze.run_model(preinit_image)

img_demo = gr.Interface(
    fn=gaze.run_model,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="image",
    live=True,
    description="Get image from webcam and estimate gaze",
)
video_demo = gr.Interface(
    fn=gaze.run_model,
    inputs=gr.Video(),
    outputs=gr.Video(label="Gaze Video"),
    description="Get video and estimate gaze",
)
demo = gr.TabbedInterface([img_demo, video_demo], ["Webcam", "Video"])
demo.launch(share=True)  # , auth=[("id", "password")])
