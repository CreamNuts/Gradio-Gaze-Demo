import bz2
import logging
import pathlib
import tempfile
from typing import Any

import cv2
import torch.hub
import torchvision.transforms as T
import yaml
from omegaconf import DictConfig, OmegaConf

from .common.face_model import FaceModel
from .common.face_model_68 import FaceModel68
from .common.face_model_mediapipe import FaceModelMediaPipe

logger = logging.getLogger(__name__)


def create_transform(config: DictConfig) -> Any:
    size = config.transform.face_size
    transform = T.Compose(
        [
            T.Lambda(lambda x: cv2.resize(x, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def get_3d_face_model(config: DictConfig) -> FaceModel:
    if config.face_detector.mode == "mediapipe":
        return FaceModelMediaPipe()
    else:
        return FaceModel68()


def download_dlib_pretrained_model() -> None:
    logger.debug("Called download_dlib_pretrained_model()")

    dlib_model_dir = pathlib.Path("~/.ptgaze/dlib/").expanduser()
    dlib_model_dir.mkdir(exist_ok=True, parents=True)
    dlib_model_path = dlib_model_dir / "shape_predictor_68_face_landmarks.dat"
    logger.debug(f"Update config.face_detector.dlib_model_path to {dlib_model_path.as_posix()}")

    if dlib_model_path.exists():
        logger.debug(f"dlib pretrained model {dlib_model_path.as_posix()} already exists.")
        return

    logger.debug("Download the dlib pretrained model")
    bz2_path = dlib_model_path.as_posix() + ".bz2"
    torch.hub.download_url_to_file(
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", bz2_path
    )
    with bz2.BZ2File(bz2_path, "rb") as f_in, open(dlib_model_path, "wb") as f_out:
        data = f_in.read()
        f_out.write(data)


def download_ethxgaze_model() -> pathlib.Path:
    logger.debug("Called _download_ethxgaze_model()")
    output_dir = pathlib.Path("~/.ptgaze/models/").expanduser()
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "eth-xgaze_resnet18.pth"
    if not output_path.exists():
        logger.debug("Download the pretrained model")
        torch.hub.download_url_to_file(
            "https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.2.2/eth-xgaze_resnet18.pth",
            output_path.as_posix(),
        )
    else:
        logger.debug(f"The pretrained model {output_path} already exists.")
    return output_path


def generate_dummy_camera_params(config: DictConfig, image) -> None:
    logger.debug("Called _generate_dummy_camera_params()")
    h, w = image.shape[:2]
    logger.debug(f"Frame size is ({w}, {h})")
    dic = {
        "image_width": w,
        "image_height": h,
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [w, 0.0, w // 2, 0.0, w, h // 2, 0.0, 0.0, 1.0],
        },
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0.0, 0.0, 0.0, 0.0, 0.0]},
    }
    config.gaze_estimator.camera_params = dic
    with open("config.yaml", "w") as fp:
        OmegaConf.save(config=config, f=fp.name)
