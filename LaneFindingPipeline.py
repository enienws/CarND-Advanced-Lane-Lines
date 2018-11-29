import numpy as np
import cv2
from CameraCalibration import ReadCameraCalibrationMatrix, UndistortImage
from ImageSourceGrabber import ImageSourceGrabber
from PerspectiveTransformer import PerspectiveTransformer
from ImageThresholder import ImageThresholder
from LaneDetector import LaneDetector
import matplotlib.pyplot as plt
from ImageSourceGrabber import ImageSourceGrabber
from VideoSourceGrabber import VideoSourceGrabber

#Read the camera calibration matrix
cameraMatrix, distCoeffs = ReadCameraCalibrationMatrix()

src_top_right = [713, 449]
src_bottom_right = [1101, 623]
src_bottom_left = [224, 623]
src_top_left = [582, 449]

# dst_top_right = [907, 449]
# dst_bottom_right = [907, 623]
# dst_bottom_left = [403, 623]
# dst_top_left = [403, 449]

#Original
# dst_top_right = [1120, 90]
# dst_bottom_right = [1120, 630]
# dst_bottom_left = [160, 630]
# dst_top_left = [160, 90]

dst_top_right = [1120, 0]
dst_bottom_right = [1120, 720]
dst_bottom_left = [160, 720]
dst_top_left = [160, 0]


def PrepareColoredThresholdedImage(thresholded_image):
    thresholded_image_left = thresholded_image.copy() * 255
    thresholded_image_right = thresholded_image.copy() * 255
    thresholded_image_left[:, int(thresholded_image.shape[1] / 2):thresholded_image.shape[1]] = 0
    thresholded_image_right[:,0:int(thresholded_image.shape[1]/2)] = 0
    colored_image = np.dstack((thresholded_image_right,np.zeros_like(thresholded_image),thresholded_image_left))
    return colored_image


def Pipeline(image):
    # Initialize perspective transformer
    perspectiveTransformer = PerspectiveTransformer(
        np.float32([src_top_right, src_bottom_right, src_bottom_left, src_top_left]),
        np.float32([dst_top_right, dst_bottom_right, dst_bottom_left, dst_top_left])
    )

    # Initialize image thresholder
    thresholder = ImageThresholder()

    # Initialize histogram
    laneDetector = LaneDetector()

    #Perform image undistortion
    undistorted_image = UndistortImage(image, cameraMatrix, distCoeffs)

    #Perform perspective transform
    perspective_transformed = perspectiveTransformer.TransformImage(undistorted_image)

    #Perform thresholding
    thresholded_image, _, _, _, _, _, _ = thresholder.Perform(perspective_transformed)

    #Calculate the histogram
    centroids = laneDetector.FindWindowCentroids(thresholded_image)
    centroids_image = laneDetector.DrawWindowCentroids(thresholded_image, centroids)
    left_poly, right_poly = laneDetector.FitAPolyline(thresholded_image, centroids)
    poly_region_image = laneDetector.DrawPolylines(undistorted_image, left_poly, right_poly)
    radius_left = laneDetector.FindRadiusOfCurvature(left_poly, 720)
    radius_right = laneDetector.FindRadiusOfCurvature(right_poly, 720)
    curvature = (radius_left + radius_right) / 2.0
    center = laneDetector.CarPosition(left_poly, right_poly)

    #Apply inverse transform on lane region
    lane_detector_region = perspectiveTransformer.InverseTransformImage(poly_region_image)

    pipeline_output = cv2.addWeighted(undistorted_image, 1, lane_detector_region, 0.3, 0)


    return pipeline_output, undistorted_image, thresholded_image, perspective_transformed, centroids_image, poly_region_image, lane_detector_region, curvature, center


if __name__ == "__main__":
    #Create the image grabber
    imageGrabber = ImageSourceGrabber("/home/engin/Documents/Projects/CarND/CarND-Advanced-Lane-Lines/test_images")
    #imageGrabber = VideoSourceGrabber("/home/engin/Documents/Projects/CarND/CarND-Advanced-Lane-Lines/project_video.mp4")
    #imageGrabber = VideoSourceGrabber("/home/engin/Documents/Projects/CarND/CarND-Advanced-Lane-Lines/challenge_video.mp4")

    imageGrabber.Init()

    counter = 0
    while imageGrabber.HasNext():
        image = imageGrabber.GrabImage()
        if image is None:
            break
        #image = cv2.imread("/home/engin/Documents/Projects/CarND/CarND-Advanced-Lane-Lines/test_images/test3.jpg")

        pipeline_output, undistorted_image, thresholded_image, \
        image, centroids_image, poly_image, \
        lane_detector_region, curvature, center = Pipeline(image)

        cv2.putText(pipeline_output, "Curvature: {0:.2f} m".format(curvature), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(pipeline_output, "Center: {0:.2f} m".format(center), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)


        cv2.imshow("ThresholdedImage", thresholded_image * 255)
        cv2.imshow("Image", image)
        cv2.imshow("Centroids", centroids_image)
        cv2.imshow("PolylineImage", poly_image)
        cv2.imshow("LaneDetectorRegion", lane_detector_region)
        cv2.imshow("Weighted", pipeline_output)

        cv2.imwrite("./test_images_out/{}.png".format(counter), pipeline_output)
        counter = counter + 1
        cv2.waitKey(33)
