import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self, srcRegion, destRegion):
        self.srcRegion = srcRegion
        self.destRegion = destRegion
        self.transformationMatrix = cv2.getPerspectiveTransform(np.array(srcRegion), np.array(destRegion))
        self.inverseTransformationMatrix = cv2.getPerspectiveTransform(np.array(destRegion), np.array(srcRegion))
        self.sharpen_kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])

    def TransformImage(self, image):
        image = cv2.warpPerspective(image, self.transformationMatrix, (1280,720))
        image_sharpen = cv2.filter2D(image, -1, self.sharpen_kernel)
        return image_sharpen

    def InverseTransformImage(self, image):
        return cv2.warpPerspective(image, self.inverseTransformationMatrix, (1280,720))

if __name__ == "__main__":
    image = cv2.imread("./images/project_video_sample_2.png")

    #Draw the quadrilateral area on the image
    #cv2.line(image, )

    src_top_right = [713, 449]
    src_bottom_right = [1101, 623]
    src_bottom_left = [224, 623]
    src_top_left = [582, 449]


    dst_top_right = [1120, 0]
    dst_bottom_right = [1120, 720]
    dst_bottom_left = [160, 720]
    dst_top_left = [160, 0]


    perspectiveTransformer = PerspectiveTransformer(#np.float32([[696, 439], [1094, 626], [179, 626], [594, 439]]),
                                                    np.float32([src_top_right, src_bottom_right, src_bottom_left, src_top_left]),
                                                    #np.float32([[1280, 0], [1280, 720], [0, 720], [0, 0]]))
                                                    np.float32([dst_top_right, dst_bottom_right, dst_bottom_left, dst_top_left]))
    image_warped = perspectiveTransformer.TransformImage(image)

    cv2.line(image, tuple(src_top_right), tuple(src_top_left), (255,0,0), 2)
    cv2.line(image, tuple(src_top_left), tuple(src_bottom_left), (255,0,0), 2)
    cv2.line(image, tuple(src_bottom_left), tuple(src_bottom_right), (255, 0, 0), 2)
    cv2.line(image, tuple(src_bottom_right), tuple(src_top_right), (255, 0, 0), 2)
    cv2.imshow("Unwarped", image)

    # cv2.line(image_warped, tuple(dst_top_right), tuple(dst_top_left), (255,0,0), 2)
    # cv2.line(image_warped, tuple(dst_top_left), tuple(dst_bottom_left), (255,0,0), 2)
    # cv2.line(image_warped, tuple(dst_bottom_left), tuple(dst_bottom_right), (255, 0, 0), 2)
    # cv2.line(image_warped, tuple(dst_bottom_right), tuple(dst_top_right), (255, 0, 0), 2)

    cv2.imwrite("./images/project_video_sample_2_warped.png", image_warped)
    cv2.imshow("Warped", image_warped)
    cv2.waitKey()

