from typing import Dict, List, Sequence, Tuple, Union

import argparse
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
from utils import get_projection_matrix, triangulate_points, write_keypoints_to_disk

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [720, 1280]

InputSpec = Union[int, str]


def run_mp(inputs: Sequence[InputSpec], projections: Sequence[np.ndarray]):
    if len(inputs) != len(projections):
        raise ValueError("Number of inputs must match number of projection matrices")
    if len(projections) < 2:
        raise ValueError("run_mp requires at least two projection matrices")

    #input video stream
    caps = [cv.VideoCapture(stream) for stream in inputs]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    #create hand keypoints detector object.
    hands = [
        mp_hands.Hands(
            min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5
        )
        for _ in inputs
    ]

    #containers for detected keypoints for each camera
    kpts_2d = [[] for _ in inputs]
    kpts_3d = []

    while True:

        #read frames from stream
        frame_data = [cap.read() for cap in caps]
        if not all(ret for ret, _ in frame_data):
            break

        frames = [frame for _, frame in frame_data]

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        for idx, frame in enumerate(frames):
            if frame.shape[1] != frame_shape[0]:
                center = frame.shape[1] // 2
                half_width = frame_shape[0] // 2
                frames[idx] = frame[:, center - half_width : center + half_width]

        # the BGR image to RGB.
        rgb_frames = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames]

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        for frame in rgb_frames:
            frame.flags.writeable = False

        results = [hand.process(frame) for hand, frame in zip(hands, rgb_frames)]

        frame_keypoints_per_camera = []
        for idx, (frame, result) in enumerate(zip(rgb_frames, results)):
            frame_keypoints = []
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    for p in range(21):
                        pxl_x = int(round(frame.shape[1] * hand_landmarks.landmark[p].x))
                        pxl_y = int(round(frame.shape[0] * hand_landmarks.landmark[p].y))
                        kpts = [pxl_x, pxl_y]
                        frame_keypoints.append(kpts)
            else:
                #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
                frame_keypoints = [[-1, -1]] * 21

            kpts_2d[idx].append(frame_keypoints)
            frame_keypoints_per_camera.append(frame_keypoints)

        #calculate 3d position
        frame_p3ds = []
        num_landmarks = len(frame_keypoints_per_camera[0]) if frame_keypoints_per_camera else 0
        for landmark_idx in range(num_landmarks):
            observations = [
                frame_keypoints_per_camera[camera_idx][landmark_idx]
                for camera_idx in range(len(projections))
            ]
            _p3d = triangulate_points(projections, observations)
            frame_p3ds.append(_p3d)

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((21, 3))
        kpts_3d.append(frame_p3ds)

        # Draw the hand annotations on the image.
        for frame in rgb_frames:
            frame.flags.writeable = True

        bgr_frames = [cv.cvtColor(frame, cv.COLOR_RGB2BGR) for frame in rgb_frames]

        for frame, result in zip(bgr_frames, results):
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for idx, frame in enumerate(bgr_frames):
            cv.imshow(f"cam{idx}", frame)

        k = cv.waitKey(1)
        if k & 0xFF == 27:
            break  #27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    for hand in hands:
        hand.close()

    kpts_2d_arrays = [np.array(cam_kpts) for cam_kpts in kpts_2d]
    return (*kpts_2d_arrays, np.array(kpts_3d))

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MediaPipe hand pose estimation on multiple camera feeds or video files."
        )
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help=(
            "Camera identifiers or video file paths. If omitted, bundled demo videos "
            "will be used."
        ),
    )
    return parser


def _coerce_sources(raw_sources: Sequence[str]) -> Tuple[List[InputSpec], List[int]]:
    """Convert CLI source inputs to OpenCV inputs and calibration identifiers."""

    inputs: List[InputSpec] = []
    camera_ids: List[int] = []

    for idx, source in enumerate(raw_sources):
        try:
            camera_id = int(source)
        except ValueError:
            camera_id = idx
            inputs.append(source)
        else:
            inputs.append(camera_id)

        camera_ids.append(camera_id)

    return inputs, camera_ids


def _validate_calibration_files(camera_ids: Sequence[int], calibration_dir: Path) -> None:
    """Ensure calibration files exist for every requested camera."""

    missing: Dict[int, List[str]] = {}
    for camera_id in camera_ids:
        required_files = [
            calibration_dir / f"c{camera_id}.dat",
            calibration_dir / f"rot_trans_c{camera_id}.dat",
        ]
        absent = [str(path) for path in required_files if not path.is_file()]
        if absent:
            missing[camera_id] = absent

    if missing:
        message_lines = ["Missing calibration files for the requested cameras:"]
        for camera_id, files in missing.items():
            message_lines.append(f"  Camera {camera_id}: {', '.join(files)}")
        raise FileNotFoundError("\n".join(message_lines))


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.sources:
        raw_sources = args.sources
    else:
        raw_sources = ["media/cam0_test.mp4", "media/cam1_test.mp4"]

    inputs, camera_ids = _coerce_sources(raw_sources)

    if len(inputs) < 2:
        parser.error("At least two camera identifiers or video sources are required.")

    try:
        _validate_calibration_files(camera_ids, Path("camera_parameters"))
    except FileNotFoundError as exc:
        parser.error(str(exc))

    projections = [get_projection_matrix(camera_id) for camera_id in camera_ids]
    keypoints = run_mp(inputs, projections)

    #this will create keypoints file in current working folder
    #for idx, cam_kpts in enumerate(keypoints[:-1]):
    #    write_keypoints_to_disk(f'kpts_cam{idx}.dat', cam_kpts)
    #write_keypoints_to_disk('kpts_3d.dat', keypoints[-1])
