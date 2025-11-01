from typing import Sequence, Union

import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
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

if __name__ == '__main__':

    input_stream1 = 'media/cam0_test.mp4'
    input_stream2 = 'media/cam1_test.mp4'

    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    #projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(
        [input_stream1, input_stream2],
        [P0, P1],
    )

    #this will create keypoints file in current working folder
    #write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    #write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    #write_keypoints_to_disk('kpts_3d.dat', kpts_3d)
