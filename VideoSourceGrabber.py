import cv2
from BaseImageGrabber import BaseImageGrabber

class VideoSourceGrabber(BaseImageGrabber):
    def __init__(self, videoPath):
        self.videoPath = videoPath
        self.videoEOF = False
        BaseImageGrabber.__init__(self)


    def Init(self):
        self.video = cv2.VideoCapture()
        self.video.open(self.videoPath)
        if self.video.isOpened() == False:
            print("Video file cannot be opened.")

    def GrabImage(self):
        retVal = self.video.grab()
        if retVal == 0:
            self.videoEOF = True
            return None
        else:
            retVal, image = self.video.retrieve()

            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(image_lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            image_lab = cv2.merge(lab_planes)
            image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)

            return image

    def HasNext(self):
        """Check whether there are unprocessed images..
        """
        return not self.videoEOF




