import numpy as np
import cv2

class ImageThresholder:

    def Morph_Open(self, image, iteration = 1, kernel=np.ones((3,3), np.uint8)):
        result = image
        for i in range(0, iteration):
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        return result


    def abs_sobel_thresh(self, derivative_x, derivative_y, abs_threshold_x=(0,255), abs_threshold_y=(0,255)):
        #Fetch the thresholds
        abs_thr_x_min, abs_thr_x_max = abs_threshold_x
        abs_thr_y_min, abs_thr_y_max = abs_threshold_y

        abs_derivative_x = np.absolute(derivative_x)
        scaled_x = np.uint8(255 * abs_derivative_x / np.max(abs_derivative_x))
        binary_output_x = np.zeros_like(scaled_x)
        binary_output_x[(scaled_x >= abs_thr_x_min) & (scaled_x <= abs_thr_x_max)] = 1

        abs_derivative_y = np.absolute(derivative_y)
        scaled_y = np.uint8(255 * abs_derivative_y / np.max(abs_derivative_y))
        binary_output_y = np.zeros_like(scaled_y)
        binary_output_y[(scaled_y >= abs_thr_y_min) & (scaled_x <= abs_thr_y_max)] = 1

        return binary_output_x, binary_output_y

    def mag_sobel_thresh(self, derivative_x, derivative_y, mag_threshold=(0,255)):
        #Fetch the thresholds
        mag_threshold_min, mag_threshold_max = mag_threshold

        # Calculate the gradient magnitude
        gradmag = np.sqrt(derivative_x ** 2 + derivative_y ** 2)

        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_threshold_min) & (gradmag <= mag_threshold_max)] = 1

        return binary_output

    def dir_sobel_thresh(self, derivative_x, derivative_y, dir_threshold=(0, np.pi/2)):
        #Fetch the thresholds
        dir_threshold_min, dir_threshold_max = dir_threshold

        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(derivative_y), np.absolute(derivative_x))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= dir_threshold_min) & (absgraddir <= dir_threshold_max)] = 1

        # Return the binary image
        return binary_output

    def hls_colorspace_thresh(self, img, h_threshold=(0, 255), s_threshold=(0, 255)):
        #Fetch the thresholds
        h_threshold_min, h_threshold_max = h_threshold
        s_threshold_min, s_threshold_max = s_threshold

        #Convert RGB image to HLS image
        hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        #Get the channels
        h_channel = hls_image[:, :, 0]
        s_channel = hls_image[:, :, 2]

        #Threshold hue channel
        binary_h = np.zeros((hls_image.shape[0], hls_image.shape[1]))
        binary_h[(h_channel > h_threshold_min) & (h_channel <= h_threshold_max)] = 1

        #Threshold saturation channel
        binary_s = np.zeros((hls_image.shape[0], hls_image.shape[1]))
        binary_s[(s_channel > s_threshold_min) & (s_channel <= s_threshold_max)] = 1

        return binary_h, binary_s

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros_like(img)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (255), 2)

        cv2.imshow("LineImg", line_img)
        return line_img

    def Perform(self, image):
        sobel_kernel = 15

        #Convert the color space
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Apply Sobel filter in both directions
        derivative_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        derivative_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        #Perform different thresholdings on the image
        abs_sobel_image_x, abs_sobel_image_y = self.abs_sobel_thresh(derivative_x, derivative_y, (25, 255), (50,255))
        abs_sobel_image_x = self.Morph_Open(abs_sobel_image_x)
        abs_sobel_image_y = self.Morph_Open(abs_sobel_image_y)

        mag_sobel_image = self.mag_sobel_thresh(derivative_x, derivative_y, (50,255))
        mag_sobel_image = self.Morph_Open(mag_sobel_image)

        dir_sobel_image = self.dir_sobel_thresh(derivative_x, derivative_y, (0.0, 0.1))
        #dir_sobel_image = self.Morph_Open(dir_sobel_image)

        #dir_sobel_image = self.dir_sobel_thresh(image_gray, derivative_x, derivative_y, (-0.3925, -0.7852))

        color_h_binary_image, color_s_binary_image = self.hls_colorspace_thresh(image, (0, 17), (90, 255))
        color_h_binary_image = self.Morph_Open(color_h_binary_image)
        color_s_binary_image = self.Morph_Open(color_s_binary_image)

        #Combine  the results
        combined = np.zeros_like(abs_sobel_image_x)
        combined[ (abs_sobel_image_x == 1) | ((color_s_binary_image == 1)) ] = 1
        combined = self.Morph_Open(combined, iteration=1)

        # rho = 1
        # theta = np.pi / 180
        # threshold = 20
        # min_line_len = 5
        # max_line_gap = 5
        # self.hough_lines(combined, rho, theta, threshold, min_line_len, max_line_gap)

        return combined, abs_sobel_image_x, abs_sobel_image_y, mag_sobel_image, dir_sobel_image, color_h_binary_image, color_s_binary_image


if __name__ == "__main__":
    #Create the thresholder
    thresholder = ImageThresholder()

    #Read the image
    image = cv2.imread("./images/project_video_sample_warped.png")

    combined, abs_sobel_image_x, abs_sobel_image_y, mag_sobel_image, dir_sobel_image, color_h_binary_image, color_s_binary_image = thresholder.Perform(image)

    cv2.imwrite("combined.jpg", combined * 255)
    cv2.imwrite("abs_sobel_x.jpg", abs_sobel_image_x * 255)
    cv2.imwrite("abs_sobel_y.jpg", abs_sobel_image_y * 255)
    cv2.imwrite("mag_sobel.jpg", mag_sobel_image*255)
    cv2.imwrite("dir_sobel.jpg", dir_sobel_image*255)
    cv2.imwrite("color_binary_h.jpg", color_h_binary_image * 255)
    cv2.imwrite("color_binary_s.jpg", color_s_binary_image * 255)


    cv2.imshow("Combined", combined*255)
    cv2.imshow("Abs_Sobel_x", abs_sobel_image_x*255)
    cv2.imshow("Abs_Sobel_y", abs_sobel_image_y*255)
    cv2.imshow("Mag_Sobel", mag_sobel_image*255)
    cv2.imshow("Dir_Sobel", dir_sobel_image*255)
    cv2.imshow("Color_Binary_H", color_h_binary_image * 255)
    cv2.imshow("Color_Binary_S", color_s_binary_image * 255)

    cv2.waitKey()