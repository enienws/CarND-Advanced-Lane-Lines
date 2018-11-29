import cv2
import numpy as np



class LaneDetector:
    def __init__(self):
        self.left_model = None
        self.right_model = None
        self.window_centroids = None
        self.left_centroids = None
        self.right_centroids = None

    def CalcHist(self, image):
        histogram = np.sum(image[int(image.shape[0]/2):,:], axis=0)
        return histogram

    def CalculatePolynomial(self, polynomial, x):
        #Calculate the polynomial equation
        A, B, C = polynomial
        return A*x*x + B*x + C

    def FindRadiusOfCurvature(self, polynomial, y):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension

        #Calculate radius of curvature for a given polynomial at given point
        A, B, C = polynomial
        return (1 + (2*A*y*ym_per_pix + B)**2)**1.5 / np.absolute(2 * A)

    def CarPosition(self, left_model, right_model):
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
        A_left, B_left, C_left = left_model
        A_right, B_right, C_right = right_model
        position = 640 # 1280 / 2
        height = 720
        l_pos = A_left * height ** 2 + B_left * height + C_left
        r_pos = A_right * height ** 2 + B_right * height + C_right
        lane_center_position = (l_pos + r_pos) / 2
        center_dist = (position - lane_center_position) * xm_per_pix
        return center_dist

    def FindRadiusOfCurvatures(self, left_model, right_model, y):
        #Calculate radius of curvatures in pixels
        left_curvature = self.FindRadiusOfCurvature(left_model, y)
        right_curvature = self.FindRadiusOfCurvature(right_model, y)
        return left_curvature, right_curvature

    def ConvertRadiusOfCurvature(self, curvature):
        return curvature

    def ConvertRadiusOfCurvatures(self, left_curvature, right_curvature):
        left_curvature_meter = self.ConvertRadiusOfCurvature(left_curvature)
        right_curvature_meter = self.ConvertRadiusOfCurvature(right_curvature)
        return left_curvature_meter, right_curvature_meter

    def DrawPolylines(self, image, left_model, right_model, window_height=120):
        vertices_right = []
        vertices_left = []

        for i in range(0, 760, 40):
            #Determine the position of left centroid
            left_centroid_x, left_centroid_y = int(self.CalculatePolynomial(left_model, i)), i
            #left_centroid_x = int(self.CalculatePolynomial(left_model, left_centroid_y))
            vertices_left.append((left_centroid_x, left_centroid_y))

        for i in range(0, 760, 40):
            #Determine the position of left centroid
            right_centroid_x, right_centroid_y = int(self.CalculatePolynomial(right_model, i)), i
            #right_centroid_x = int(self.CalculatePolynomial(right_model, right_centroid_y))
            vertices_right.append((right_centroid_x, right_centroid_y))


        #Draw the poly region
        polyline_region_image = np.zeros_like(image)
        vertices = np.array([vertices_left + list(reversed(vertices_right))], dtype=np.int32)
        cv2.fillPoly(polyline_region_image, vertices, (0,255,0))

        #Draw left lane
        for i in range(0, len(vertices_left)-1):
            begin = vertices_left[i]
            end = vertices_left[i+1]
            cv2.line(polyline_region_image, begin, end, (255,0,0), 20)
        #Draw right lane
        for i in range(0, len(vertices_right)-1):
            begin = vertices_right[i]
            end = vertices_right[i+1]
            cv2.line(polyline_region_image, begin, end, (0,0,255), 20)

        #polyline_region_image = np.dstack((np.zeros_like(image), polyline_region_image, np.zeros_like(image)))

        return polyline_region_image

    def FitAPolyline(self, image, window_centroids, window_height=120):
        #List to hold centroid positions
        left_centroids = []
        right_centroids = []

        total_levels = int(image.shape[0] / window_height)
        current_level = total_levels - 1
        for window_centroid in window_centroids:
            #Determine the position of left centroid
            if window_centroid[0] != 0:
                left_centroid = [int(window_centroid[0]), int(current_level * window_height + window_height/2)]
                left_centroids.append(left_centroid)

            #Determine the position of right centroid
            if window_centroid[1] != 0:
                right_centroid = [int(window_centroid[1]), int(current_level * window_height + window_height/2)]
                right_centroids.append(right_centroid)

            #Iterate the level
            current_level = current_level - 1

        left_centroids = np.array(left_centroids)
        self.left_centroids = left_centroids = left_centroids.transpose()

        self.left_model = left_polyfit = np.polyfit(left_centroids[1,:], left_centroids[0,:], 2)

        right_centroids = np.array(right_centroids)
        self.right_centroids = right_centroids = right_centroids.transpose()
        self.right_model = right_polyfit = np.polyfit(right_centroids[1,:], right_centroids[0,:], 2)

        return left_polyfit, right_polyfit

    def DrawWindowCentroids(self, image, window_centroids, window_width=50, window_height=120):
        window_image = np.zeros_like(image)
        total_levels = int(image.shape[0] / window_height)
        current_level = total_levels - 1
        for window_centroid in window_centroids:
            left_rect_tl = (int(window_centroid[0]-window_width/2), int(current_level * window_height))
            cv2.rectangle(window_image, left_rect_tl, (left_rect_tl[0] + window_width, left_rect_tl[1] + window_height),
                          (255), cv2.FILLED)

            right_rect_tl = (int(window_centroid[1]-window_width/2), int(current_level * window_height))
            cv2.rectangle(window_image, right_rect_tl, (right_rect_tl[0] + window_width, right_rect_tl[1] + window_height),
                          (255), cv2.FILLED)
            current_level = current_level - 1

        return window_image

    def FindWindowCentroids(self, image, window_width=50, window_height=120, margin=100):
        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))


        total_levels = int(image.shape[0] / window_height)
        current_level = total_levels - 1
        # Go through each layer looking for max pixel locations
        for level in range(1, total_levels):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(
                image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)

            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, image.shape[1]))
            left_convolved_signal = conv_signal[l_min_index:l_max_index]
            #print("left: {}".format(np.sum(left_convolved_signal)))
            l_append = True
            r_append = True
            if np.sum(left_convolved_signal) > 5000:
                l_center = np.argmax(left_convolved_signal) + l_min_index - offset
            else:
                # do not place a window
                l_append = False
                pass

            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, image.shape[1]))
            right_convolved_signal = conv_signal[r_min_index:r_max_index]
            #print("right: {}".format(np.sum(right_convolved_signal)))
            if np.sum(right_convolved_signal) > 5000:
                r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            else:
                # do not place a window
                r_append = False
                pass


            # Add what we found for that layer
            if (r_append and l_append) == True:
                window_centroids.append((l_center, r_center))
            elif r_append == False and l_append == True:
                window_centroids.append((l_center, 0))
            elif r_append == False and l_append == True:
                window_centroids.append((0, r_center))
            elif (r_append and l_append) == False:
                window_centroids.append((0,0))

        self.window_centroids = window_centroids

        return window_centroids