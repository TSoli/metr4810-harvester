# Controller

This directory contains the software designed to run on the centralised
controller.

## Getting Started

1. Calibrate the camera that is being used. Ensure that the intrinsic camera
   parameters will not change during recording (e.g due to auto focus).
   Instructions for camera calibration with ROS2 can be found
   [here](https://docs.nav2.org/tutorials/docs/camera_calibration.html). Some
   example calibration files for a Google Pixel 8 can be found in the
   `calibration/` directory (note that parameters may vary slightly even for the
   same camera model).
2. Run the camera. A convenient method to use a smartphone is to download
   [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam&pcampaignid=web_share)
   and tether the smartphone to the computer for low delay. Note the IP address
   in the app and open it in a browser. Open focus mode, ensure it is manually
   set and choose the same value as specified when the camera was calibrated.
   This information is recorded for the example calibration in
   `calibration/readme.txt`.
3. For testing run the localisation script

```sh
python localisation.py [device] [cam_params] --size <marker_dict> --square <marker_size>
```

E.g for the IP Webcam setup, the device would be the IP address followed by
`/video`. For example, for marker size 90mm using 4x4_50 dictionary

```sh
python localisation.py "https://192.168.0.12:8080/video" calibration/ost.yaml --size DICT_4X4_50 --square 0.09
```
