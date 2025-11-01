**Real time 3D hand pose estimation using MediaPipe**

This is a demo on how to obtain 3D coordinates of hand keypoints using MediaPipe and two calibrated cameras. Two cameras are required as there is no way to obtain 3D coordinates from a single camera. Check here: [stereo calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate) for a calibration package. Also my blog post on how to stereo calibrate two cameras: [link](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html). Alternatively, follow the camera calibration at Opencv documentations: [link](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html). If you want to know some details on how this code works, take a look at my accompanying blog post here: [link](https://temugeb.github.io/python/computer_vision/2021/06/27/handpose3d.html).

![input1](media/output_kpts.gif "input1") ![input2](media/output2_kpts.gif "input2")
![output](media/fig_0.gif "output")

**MediaPipe**
Install mediapipe in your virtual environment using:
```
pip install mediapipe
```

**Requirements**
```
Mediapipe
Python3.8
Opencv
matplotlib
```

**Usage: Getting real time 3D coordinates**

Follow the steps below to run the demo or adapt it for your own stereo camera rig.

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # or install mediapipe, opencv-python, numpy, matplotlib manually
   ```

2. **Prepare calibration data**
   - Each camera requires an intrinsic calibration file (``camera_parameters/cN.dat``) and an extrinsic rotation/translation file (``camera_parameters/rot_trans_cN.dat``).
   - Use the [stereo calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate) tool or the [OpenCV calibration tutorial](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html) to capture chessboard images, compute the stereo parameters, and export them in the expected ``.dat`` format.
   - Match the numbering of your calibration files (``c0``, ``c1``, …) with the order you will pass the camera sources to the script.

3. **Run the real-time or prerecorded demo**
   - The repository ships with two sample videos and matching calibration files. To replay them and verify your setup, run:
     ```bash
     python handpose3d.py
     ```
   - To use live webcams (for example, devices ``0`` and ``1``) run:
     ```bash
     python handpose3d.py 0 1
     ```
   - You can mix and match video files and camera identifiers. Provide one argument per camera in the same order as their calibration files, e.g.:
     ```bash
     python handpose3d.py cam0.mp4 cam1.mp4 cam2.mp4
     ```

4. **Inspect the output**
   - During execution each frame’s 3D coordinates are appended to ``frame_p3ds`` and persisted to ``kpts_3d.dat`` when you exit with the **ESC** key.
   - Missing keypoints are reported as ``(-1, -1, -1)``. Long sessions will grow the lists ``kpts_3d``, ``kpts_cam0``, ``kpts_cam1`` indefinitely; remove their ``append`` calls if you need a constant-memory streaming setup.
   - After you exit, three ``.dat`` files containing 2D and 3D trajectories are saved in the project root.

**Usage: Viewing 3D coordinates**

To visualize previously recorded coordinates, run:
```bash
python show_3d_hands.py
```
This opens a Matplotlib window that replays the 3D hand skeleton over time.

**Testing and troubleshooting**

- The automated test suite validates stereo triangulation logic. Execute it with:
  ```bash
  python -m pytest
  ```
- If no hand keypoints are detected, ensure adequate lighting and that hands are within both camera frames. Adjust MediaPipe confidence thresholds in ``handpose3d.py`` if necessary.
- For calibration issues, verify that all ``camera_parameters`` files correspond to the cameras in use and that the cameras are rigidly mounted to maintain the calibrated baseline.

