The Project
---

The goals / steps of this project are the following:
1. Import all the packages opencv, matplotlib, numpy and moviepy
2. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Camera calibration is calculated with the function calibrate_camera where all the images from 'camera_cal/calibration*.jpg' are passed and the distortion coeeficients are calculated on the basis of all the images.

Each image is read using cv2.imread and then converted to grayscale using cv2.cvtColor. 

Chessboard corners are found using findChessboardCorners and if the corners are present then corners of the image are displayed using drawChessboardCorners.

The image is then calibrated using cv2.calibrateCamera and undistorted using cv2.undistort. The calibration matrix and distortion coefficients are returned.

[image1]: ./camera_cal/calibration18.jpg

[image2]: ./output_images/calibration.jpg


* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


