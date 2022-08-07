import datetime
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from .common import Face, FacePartsName, Visualizer
from .gaze_estimator import GazeEstimator
from .utils import get_3d_face_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord("q")}

    def __init__(self, config: DictConfig):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera, face_model_3d.NOSE_INDEX)

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_template_model = self.config.demo.show_template_model

    def run_model(self, input):
        if isinstance(input, str):
            # new_path = self._process_video(input)
            return "/home/jiuk/Demo/video01_gaze.mp4"  # new_path
        else:
            self._process_image(input)
            return self.visualizer.image

    def _process_image(self, image) -> None:
        undistorted = cv2.undistort(
            image,
            self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients,
        )
        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)

    def _create_capture_and_writer(self, video_path):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        # frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_path = video_path.with_suffix(".mp4").with_stem(video_path.stem + "_gaze")
        wrt = cv2.VideoWriter(
            str(new_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (frameWidth, frameHeight),
        )
        return cap, wrt, new_path

    def _process_video(self, video_path) -> None:
        cap, wrt, new_path = self._create_capture_and_writer(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self._process_image(frame)
            wrt.write(self.visualizer.image)
        cap.release()
        wrt.release()
        return str(new_path)

    def _create_output_dir(self) -> Optional[Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime("%Y%m%d_%H%M%S")

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler("XYZ", degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(
            f"[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, "
            f"roll: {roll:.2f}, distance: {face.distance:.2f}"
        )

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks, color=(0, 255, 255), size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d, color=(255, 0, 525), size=1)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == "MPIIGaze":
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(f"[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}")
        elif self.config.mode in ["MPIIFaceGaze", "ETH-XGaze"]:
            self.visualizer.draw_3d_line(face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f"[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}")
        else:
            raise ValueError
