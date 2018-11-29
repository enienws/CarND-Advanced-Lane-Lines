import cv2
import os
import numpy as np

#corners = 9,5

#An auxillary function for generating file names for undistorted images. Implemented for testing purposes.
def GenerateChessCornersFileName(filename, append):
    filename_splitted = filename.split(".")
    filename_generated = filename_splitted[0] + "_{}.".format(append) + filename_splitted[1]
    return filename_generated

#Read the images in a given directory.
def ReadImages(dir_path):
    images = []
    for file in os.listdir(dir_path):
        if file.endswith(".jpg"):
            img_path = os.path.join(dir_path, file)
            image = cv2.imread(img_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append((img_path, image, image_gray))

    return images

#Finds the chess board corners aka shape points.
def FindChessboardCorners(images):
    corners = []
    for image_path, image, image_gray in images:
        #Find the given pattern in chessboard image
        ret, pts = cv2.findChessboardCorners(image_gray, (9,6))

        #Check whether the given pattern is found successfully.
        if ret == 0:
            print("Problem on finding chessboard corners: {}...".format(image_path))
        else:
            #Append the found corners to the list
            corners.append((image_path, image, pts))
    return corners

#Draws the found shape corners implemented for testing purposes.
def DrawChessboardCorners(corners):
    for image_path, image, pts in corners:
        image_clone = image.copy()
        cv2.drawChessboardCorners(image_clone, (9,6), pts, True)
        filename = GenerateChessCornersFileName(image_path, "corners")
        cv2.imwrite(filename, image_clone)

#Finds the calibration matrix using object and image points.
def FindCalibrationMatrix(objpoints, imgpoints, imagesize):
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280,720), None, None)
    np.savetxt("cameraMatrix.txt", cameraMatrix)
    np.savetxt("distCoeffs.txt", distCoeffs)
    return (cameraMatrix, distCoeffs, rvecs, tvecs)

# Undistorts the given image.
def UndistortImage(image, cameraMatrix, distCoeffs):
    undist_image = cv2.undistort(image, cameraMatrix, distCoeffs)
    return undist_image

# Undistorts the given image.
def UndistortImageFromFile(path, cameraMatrix, distCoeffs):
    image = cv2.imread(path)
    undist_image = cv2.undistort(image, cameraMatrix, distCoeffs)
    return undist_image

#Simply undistort a list of images by using calibration matrix by using UndistortImage method.
def UndistortImages(images, cameraMatrix, distCoeffs):
    for img_path, _, image in images:
        undist_image = UndistortImage(image, cameraMatrix, distCoeffs)
        path = GenerateChessCornersFileName(img_path, "undist")
        cv2.imwrite(path, undist_image)

#Reads the saved calibration matrix and coefficients from disk.
def ReadCameraCalibrationMatrix():
    cameraMatrix = np.loadtxt("cameraMatrix.txt")
    distCoeffs = np.loadtxt("distCoeffs.txt")
    return cameraMatrix, distCoeffs

# if __name__ == "__main__":
#     #Create object point
#     objpt_template = np.zeros((9*6,3), np.float32)
#     objpt_template[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
#
#     #Read chessboard images
#     images = ReadImages("camera_cal")
#
#     #Find the chessboard corners
#     corners = FindChessboardCorners(images)
#     imgpts = list(zip(*corners))[2]
#
#     #Create obj points vector which is as same length as imgpts
#     objpts = []
#     for i in range(0,len(imgpts)):
#         objpts.append(objpt_template)
#
#     #Draw chessbooard corners
#     #DrawChessboardCorners(corners)
#
#     #Find calibration matrix
#     cameraMatrix, distCoeffs, rvecs, tvecs = FindCalibrationMatrix(objpts, imgpts, images[0][1])
#
#     #Undistort all sample images
#     UndistortImages(images, cameraMatrix, distCoeffs)

if __name__ == "__main__":
    matrix, coeffs = ReadCameraCalibrationMatrix()
    undist = UndistortImageFromFile("/home/engin/Documents/Projects/CarND/AdvancedLaneFinding/test_images/solidWhiteCurve.jpg", matrix, coeffs)
    cv2.imwrite("/home/engin/Documents/Projects/CarND/AdvancedLaneFinding/test_images/solidWhiteCurve_undist.jpg", undist)