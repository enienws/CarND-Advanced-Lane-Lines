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
            return image

    def HasNext(self):
        """Check whether there are unprocessed images..
        """
        return not self.videoEOF




