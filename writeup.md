## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/calibration3.jpg  "Distorted"
[image2]: ./writeup_images/calibration3_undist.jpg  "Undistorted"

[image3]: ./writeup_images/solidWhiteCurve.jpg  "Real Distorted"
[image4]: ./writeup_images/solidWhiteCurve_undist.jpg  "Real Undistorted"

[image5]: ./writeup_images/arctan.png  "Arctan"

[image6]: ./writeup_images/morph_open.png  "Morph open"

[image7]: ./writeup_images/abs_sobel_x.jpg  "AbsSobelX"
[image8]: ./writeup_images/abs_sobel_y.jpg  "AbsSobelY"
[image9]: ./writeup_images/dir_sobel.jpg  "DirSobel"

[image10]: ./writeup_images/color_binary_h.jpg  "ColorBinaryH"
[image11]: ./writeup_images/color_binary_s.jpg  "ColorBinaryS"
[image12]: ./writeup_images/mag_sobel.jpg  "MagSobel"

[image13]: ./writeup_images/combined.jpg  "Combined"

[image14]: ./writeup_images/project_video_sample.png  "Unwarped"
[image15]: ./writeup_images/project_video_sample_warped.png  "Warped"

[image16]: ./writeup_images/centroid_find_thresholded.png  "Thresholded"
[image17]: ./writeup_images/centroid_find_wrong.png  "Centroid Wrong"

[image18]: ./writeup_images/pipeline_binary.png  "Pipeline Binary"
[image19]: ./writeup_images/pipeline_centroid.png  "Pipeline Centroid"

[image20]: ./writeup_images/pipeline_polylines.png  "Pipeline Polylines"
[image21]: ./writeup_images/pipeline_renderpoly.png  "Pipeline Polylines Inverse Transformed"

[image22]: ./writeup_images/pipeline_final.png  "Example Output"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera calibration part is implemented in python file called CameraCalibration.py

This module is a standalone model and responsible to find a calibration matrix for given chessboard images for the camera under consideration. Once finding the calibration matrix, it is written to a file in order for other modules to use. The mentioned files can be controlled by checking cameraMatrix.txt and distCoeffs.txt file which referring to camera calibration matrix and distortion coefficients respectively. 

First of all object points vector is constructed. Points are given in object's  world coordinate system according to number of corners since we don't have any chance to measure the edges of boxes. The points are in the form of triplets (x,y,z) where z always equals to zero due to the assumption chessboard is fixed at a plane z=0

Secondly ReadImages method is implemented which is responsible for reading chess board images from disk. This method simply reads the images and returns a vector which holds the read images. 

Thirdly FindChessboardCorners method is implemented. This method simply calls OpenCV's findChessboardCorners function in order to find the chessboard corners and returns the corners' position in image which are actually shape points. 

Finally FindCalibrationMatrix method is implemented which finds the camera matrix and distortion coefficients by using object and shape points. OpenCV's cv2.calibrateCamera() function is used. Upon calculating the matrices and coefficients method simply writes these data to disk. This way, lane finding pipeline can read distortion matrix from disk to undistort images.

Other axuillary methods are implemented in CameraCalibration module. Let me briefly introduce them:
1. ReadCalibrationMatrix -> Reads the saved calibration matrix and coefficients from disk. 
2. UndistortImages -> Simply undistort a list of images by using calibration matrix by using UndistortImage method.
3. UndistortImage -> Undistorts the given image. cv2.undistort() function is used.
4. DrawChessboardCorners -> Draws the found shape corners implemented for testing purposes.
5. GenerateChessCornersFileName -> An auxillary function for generating file names for undistorted images. Implemented for testing purposes. 
6. UndistortImageFromFile -> This method simply  read an image from disk and undistort it. Implemented for testing purposes. 

Below image shows undistorted and distorted images:
![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
![alt text][image4]

The camera matrix and distortion coefficients are read from disk as explained in previous section. Then undistortion is performed using the method UndistortImageFromFile using CameraCalibration module. 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

All related functions here are implemented in ImageThresholder module in ImageThresholder.py file. 

Nearly all the methods in the lectures are implemented and different combinations of these images are tested to obtain the accurate results. 

Below a list of implemented methods and brief explanations can be found:

1. abs_sobel_thresh -> This method takes the absolute value of the given image and then thresholds it. Two sobel filter applied images (both sobel_x and sobel_y filter) are inputs of this function.

2. mag_sobel_thresh -> This method simply calculates the magnitude of change from given two sobel images: sobel_x and sobel_y. The magnitude is calculated by using the following formula: square_root (x^2 + y^2). The the found magnitudes are normalized in a range of [0, 255]. Then the normalized image is thresholded by using the input thresholds. 

3. dir_sobel_thresh -> This method simply calculates the direction from given two sobel images: sobel_x and sobel_y. The direction is calculated by using the following formula: arctangent(y, x). This function outputs values between 0 and pi/2 since absolute value of both sobel_x and sobel_y input are taken. The following graph explains this best.
![alt text][image5]

4. hls_colorspace_thresh -> This method converts the RGB colorspace to HLS colorspace. HLS color space consists of Hue Lightness and Saturation colorspace. Saturation channel carries rich information about colorness and can be used in to threshold image according to the color information. After extracting different channels from the given image, hue and saturation channels are thresholded by using the given thresholds.

These methods are helper methods and the actual thresholding pipeline is implemented in Perform method. First of all this method converts given colored image to grayscale color space in order to apply two different sobel filters. One sobel filter captures the changes in x direction and the other one captures changes in y direction. After sobel images are calculated, these images are fed to helper methods to be processed in different ways and to be thresholded. All of these methods return [0,1] binary images which can be used in logical operations. 

After binary images are obtained, "morphological open" operation is applied in order to get rid of noise in binary images. Below image shows an Open operation. More information can be found below in wikipedia pages:
[Erosion](https://en.wikipedia.org/wiki/Erosion_(morphology)) 
[Dilation](https://en.wikipedia.org/wiki/Dilation_(morphology)) 
[Opening](https://en.wikipedia.org/wiki/Opening_(morphology)) 

![alt text][image6]

After some series of experiments I find the best combination required to solve the given problem. The combination is:
OR(Thresholded Absolute Value Sobel_X Image, Thesholded Saturation Channel Image )

Below in the images shows different operations and results:

|![alt text][image7]|![alt text][image8]|![alt text][image9]|
|:---:|:---:|:---:|
| Absolute Sobel X | Absolute Sobel Y | Direction Sobel |


|![alt text][image10]|![alt text][image11]|![alt text][image12]|
|:---:|:---:|:---:|
| Thresholded Hue Channel | Thresholded Saturation Channel | Magnitude Sobel |

And finally the mentioned combination:
![alt text][image13]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

All perspective transformation logic is handled in module called PerspectiveTransformer implemented in PerspectiveTransformer.py. 

PerspectiveTransformer class is initiliazed by source and destination points. By calling the constructor transformation and inverse transformation matrices are calculated and keep in member variables of the class. Methods of these class, TransformImage and InverseTransformImage, use the calculated matrices to transform the given images. 

To select appropriate source points an another small python module is implemented in PointSelectionApplication.py file. This application simple captures the mouse clicks and outputs the x,y positions of the mouse clicks. 

Destination points are calculated as generating 1/8th portion of image and 7/8th portion of the image in horizontal direction. The whole height of the image is used in vertical direction. Following points are used for perspective transformation:

| Position        | Source        | Destination   | 
|:-------------:|:-------------:|:-------------:| 
| Top Right      | 713, 449      | 1120, 0        | 
| Bottom Right      | 1101, 623      | 1120, 720      |
| Bottom Left      | 224, 623     | 160, 720      |
| Top Left      | 582, 449      | 160, 0        |

Below one can find example input and output images:

|![alt text][image14]|![alt text][image15]|
|:---:|:---:|
| Unwarped | Warped |


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Finding lane lines and fitting a polynomial to them is implemented in a module called LaneDetector which is implemented in LaneDetector.py file. 

LaneDetector class simply uses a moving window and extracts histograms from the moving window in order to fit lane windows to image. 

This module has its pipeline implemented as follows (this pipeline is used in LaneFindingPipeline.py file between lines 69-73)
1. FindWindowCentroids
2. DrawWindowCentroids
3. FitAPolyline
4. DrawPolylines

Now let me give you detailed information on these functions:
1. FİndWindowCentroids: This method first of all finds the beginning of lanes by placing a two windows bottom quarter of the image. One window covers the left half and the other one covers the right half. After finding the starting point of the lanes method simply start to iterate from bottom of image to top of image. For each iteration, loop determines in which location a lane is centered. Center of lanes are found by a convolution operation, a RoI from image (RoI is determined by the sliding window) and a window template is convolved in order to determine whether there exists a lane or not. 

This implementation is similar to that of in lectures. However I made some important changes in implementation. I realized that when sum of output of convolution operation is below a threshold  the sliding window starts to shift either left or right. This situation generally occurs when there are dashed lanes. 

Below image explains the mentioned situation well, note that in wrongly placed centroids image upper right centroid is placed well.

|![alt text][image16]|![alt text][image17]|
|:---:|:---:|
| Binary Image | Wrongly Placed Centroids |

In order to cope with this situation a threshold mechanism is implemented, so that if convolved signal's sum is below a defined threshold a centroid is not placed at all. This way polyline fitting works better since wrong points make polylinefitter generate wrong polynomials. 

2. DrawWindowCentroids: This method is not required for a lane detector pipeline, it was just implemented to debug the system in order to see whether centroids are placed accordingly to given binary thresholded image. Below image shows an example:

|![alt text][image18]|![alt text][image19]|
|:---:|:---:|
| Pipeline Binary | Pipeline Centroid |

3. FitAPolyline: This function fits two polynomials for left and right lanes. Previously found centroid positions are used for fitting. Numpy's polyfit() function is used. 

4. DrawPolylines: This method uses two polynomials in order to draw lane boundary. First of all boundary points are obtained for y values between interval: [0, 720]. In every iteration y value is incremented 40 pixels. This way x coordinates are obtained for provided y values. After obtaining values  from polynomials a polynomial is drawn. This polynomial will be used within the pipeline to drawn the lane region by using the PerspectiveTransformed module, specified before in this document. Below image shows an example output:

|![alt text][image20]|![alt text][image21]|
|:---:|:---:|
| Pipeline Polylines | Pipeline Polylines Inverse Transformed |


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature and car position is also calculated in LaneDetector module. There are two related methods:

1. FindRadiusOfCurvature() -> This method calculates the radius of curvature for given polynomial. For pixels to meters conversion the standard value 30 / 720 is used. Radius of curvature is calculated with following formula:
(1 + (2*A*y*scale + B)^2)^(3/2) / abs(2*A)
where A,B,C is multipliers in polynomial of form:
x = Ay^2 + By + C

2. CarPosition() -> Car position is typically the position of the car. Simply left-most and right-most points are found by using the left and right polynomials and then the mean value gives the position of car. For pixels to meters conversion 3.7 / 700 scale value is used. 

The code is between lines #23 and #44 in LaneDetector.py file. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After LaneDetector module is calculated the polylines and lane region polynomial, the original image and lane region polynomial is added together to create the output image. The addition is performed by using OpenCV's addWeighted function. Related code is in lines #79 and #80. 
**undist_plus_thresholded = cv2.addWeighted(undistorted_image, 1, colored_thresholded_image, 1.0, 0)
pipeline_output = cv2.addWeighted(undist_plus_thresholded, 1, lane_detector_region, 0.3, 0)**

Below image shows an example output:

![alt text][image22]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Actually the accuracy of pipeline highly depends on quality of binary thresholded image. If this image well constructed, lane finding by using a sliding window method works well. However if binary image cannot be constructed well, lane finding pipeline works bad. 

Binary image is being constructed by thresholding different binary images, ie. sobel_x, sobel_y, sobel_magnitude, etc. However thresholding mechanism may sometimes work very bad in different conditions. 

In other words, determined thresholds may not work with changing conditions, ie. changing the lightness of environment. So adaptive thresholding methods should be used for changing environmental changes. 

In project video, one should notice that, system starts to perform a little badly in changing cotings of asphalt. 

I have performed too many tests in order to find the best threshold set for the given problem but after a while this work seems unnecessary since finding such a threshold set is impossible. As I have mentioned an adaptive thresholding mechanism can  be implemented or a machine learning model should be used in order to determine lane lines. 