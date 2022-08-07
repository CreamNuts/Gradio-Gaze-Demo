import logging
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig
from timm.models import load_checkpoint

from .common import Camera, Face, FacePartsName
from .head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from .models import create_bucketizer, create_model
from .utils import create_transform, get_3d_face_model

logger = logging.getLogger(__name__)


class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, config: DictConfig):
        self._config = config

        self._face_model3d = get_3d_face_model(config)

        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(config.gaze_estimator.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config)
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera,
            self._normalized_camera,
            self._config.gaze_estimator.normalized_camera_distance,
        )
        self._gaze_estimation_model = self._load_model()
        self._gaze_bukcetizer = create_bucketizer(config)
        self._transform = create_transform(config)

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        load_checkpoint(model, self._config.gaze_estimator.checkpoint)
        model.to(torch.device(self._config.device))
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        self._face_model3d.estimate_head_pose(face, self.camera)
        self._face_model3d.compute_3d_pose(face)
        self._face_model3d.compute_face_eye_centers(face, self._config.mode)
        self._head_pose_normalizer.normalize(image, face)
        self._run_gaze_model(face)

    @torch.no_grad()
    def _run_gaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        N, D = prediction.shape
        # (N, D) -> (2N, D/2) -> (2N, 1) -> (N, 2)
        prediction = prediction.reshape(-1, D // 2).argmax(-1).reshape(N, -1)
        prediction = self._gaze_bukcetizer.unbucketize(prediction)
        prediction = prediction.cpu().numpy()
        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()
