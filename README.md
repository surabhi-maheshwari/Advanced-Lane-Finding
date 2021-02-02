## CarND-Advanced-Lane-Lines
The motive of this project is to detect and highlight lane on a video of images from a front facing camera mounted on a vehicle.


## Project Structure

- `camera_cal/` Directory with calibration images
- `test_images/` Directory with test images
- `output_images/` Directory with test output images
- `project_video.mp4/` Input project video'
- `solution_video.mp4` Output project video
- `Advanced_Lane_Finding.ipnyb` Jupyter notebook with all the project code and example images
- `README.md` Projecte writeup 


Example output video is as follows:

[image16]: ./output_images/output.gif
Output Video
![alt text][image16]

## Steps
The goals / steps of this project are the following:
1. Import all the packages opencv, matplotlib, numpy and moviepy

2. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Camera calibration is calculated with the function calibrate_camera where all the images from 'camera_cal/calibration*.jpg' are passed and the distortion coeeficients are calculated on the basis of all the images.

Each image is read using cv2.imread and then converted to grayscale using cv2.cvtColor. 

Chessboard corners are found using findChessboardCorners and if the corners are present then corners of the image are displayed using drawChessboardCorners.

The image is then calibrated using cv2.calibrateCamera and undistorted using cv2.undistort. The calibration matrix and distortion coefficients are returned.

```
def calibrate_camera(images):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.



    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

            #calculate camera calibration matrix
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
            dist_corr_img = cv2.undistort(img, mtx, dist, None, mtx)

    return ret, mtx, dist, rvecs, tvecs
```

3. Apply a distortion correction to raw images.

[image1]: ./camera_cal/calibration18.jpg "Calibration 18"

[image2]: ./output_images/calibration.jpg "Calibration"


Original Image
![alt text][image1] 

Undistorted Image
![alt text][image2]



4. Use color transforms, gradients, etc., to create a thresholded binary image.


[image3]: ./output_images/original.jpg
Original Image
![alt text][image3]

a. Image with x and y directional gradients:

[image4]: ./output_images/gradient_x.jpg 
Gradient x image
![alt text][image4]

[image5]: ./output_images/gradient_y.jpg
Gradient magnitude image
![alt text][image5]

```
def apply_sobel_operator(img, orient, sobel_kernel = 3, threshold = (0, 255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1

    return binary_output
```

b. Image with gradient magnitude

[image6]: ./output_images/magnitude.jpg
Gradient y image
![alt text][image6]

```
def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    sobel_image = np.zeros_like(gradmag)
    thresh_min, thresh_max = mag_thresh
    sobel_image[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1
    return sobel_image
```

c. Image with gradient direction 

[image7]: ./output_images/direction.jpg
Gradient direction image
![alt text][image7]


```
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    gradmag = np.arctan2(abs_sobely, abs_sobelx)
    sobel_image = np.zeros_like(gradmag)
    sobel_image[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return sobel_image
```

d. Image with color threshold

[image8]: ./output_images/color.jpg
Color threshold image
![alt text][image8]

```
def color_threshold(img, thresh=(170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary
```

e. Combined magnitude, gradient direction, color and sobel x and y images:

[image9]: ./output_images/combined.jpg
Combined image
![alt text][image9]

```
def combined_threshold(gradient_x, gradient_y, mag_threshold, dir_threshold, col_threshold):
    combined = np.zeros_like(dir_threshold)
    combined[((gradient_x == 1) & (gradient_y == 1)) | ((mag_threshold == 1) & (dir_threshold == 1)) | (col_threshold == 1)] = 1

    return combined
```


5. Apply a perspective transform to rectify binary image ("birds-eye view")
The next step is to convert the image to the birds-eye view.

This is done using perspective transform and the function used is cv2.getPerspectiveTransform() . Create a trapezoid on the basis of the image and pass the source and destination cordinates to the cv2.getPerspectiveTransform() function.

This function retures the transformed matrix M and it also return Minv which is inverse Matrix if the destination is passed as the first argument and src as the second.

[image10]: ./output_images/transform.jpg
Image with trapezoid drawn on the lane
![alt text][image10]


[image11]: ./output_images/warped_image.jpg
Warped image (bird's eye view)
![alt text][image11]


```
def perspective_transform(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[280,  700],  # Bottom left
     [595,  460],  # Top left
     [725,  460],  # Top right
     [1125, 700]])
    dst = np.float32([[250,  720],  # Bottom left
     [250,    0],  # Top left
     [1065,   0],  # Top right
     [1065, 720]])
    
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, src, dst, Minv
```

6. Detect lane pixels and fit to find the lane boundary.

Create a histogram of the lower half of the image and find the mid point of the image on the basis of histograme.shape[0] //2 

Then from left which is max of the histogram before the mid, and the right which is max of histogram after the mid point, start identifying lane lines in a window until the end of the lane. This process is called Sliding Window.


Fit a second order polynomial to each using `np.polyfit` and create windows on top of the warped image as follows:

[image12]: ./output_images/fit_polynomial.png
Sliding Window
![alt text][image12]

Next step is to create a highlighted window around the lanes and fit the lane boundary as shown in following image:

[image17]: output_images/fit_lane_boundary.png
Highlighted lane 
![alt text][image17]


7. Determine the curvature of the lane and vehicle position with respect to center.

Find the radius of curvature of the lanes based on the formula 

[image13]: output_images/radius_of_curvation.JPG
![alt text][image13]

Also, find the offset of the car on the basis of image center position and the lane center position.

8. Warp the detected lane boundaries back onto the original image.

Use cv2.fillPoly to highlight the lane on the image
The warped image with lane boundaries is as follows:
  
[image14]: output_images/lane_boundary.jpg
Detected lane
![alt text][image14]
  

9. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Show the horizontal offset and the lanes curvature on the image as follows:

[image15]: output_images/lane.jpg
Numerical estimation on the detected lane
![alt text][image15]

```
def vehicle_position(img, leftx, rightx):
    xm_per_pix = 3.7/700
    hor_pos = img.shape[1] / 2
    lane_pos = (leftx[-1] + rightx[-1]) / 2
    offset_x = (hor_pos - lane_pos) * xm_per_pix
    return offset_x
```

10. Run the pipeline for a set of image to generate the video saved as "project_video_Solution.mp4" on the main directory.
```
from moviepy.editor import VideoFileClip
from IPython.display import HTML

input_video = './project_video.mp4'
output_video = './solution_video.mp4'

## You may uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip(input_video).subclip(0,5)
clip1 = VideoFileClip(input_video)

# Process video frames with our 'process_image' function
process_image = ProcessImage('./camera_cal/calibration*.jpg')

white_clip = clip1.fl_image(process_image)

%time white_clip.write_videofile(output_video, audio=False)


```

## Discussion

I really enjoyed working on this project, though it was little challenging but it was fun.

The drawbacks of the above process can be as follows:

a. It might not work in very curvy roads as in the harder challenging video.

b. The pipeline works really well near the camera but there are minor failures on the edges or areas far from the camera. It sometimes cannot completely detect the lane perfectly and might go out of the bounds because of the trapezoid source and destination vertices.


