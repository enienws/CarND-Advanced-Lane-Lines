import os
import cv2
from BaseImageGrabber import BaseImageGrabber

class ImageSourceGrabber(BaseImageGrabber):
    def __init__(self, imageSourceFolder):
        self.imageSourceFolder = imageSourceFolder
        self.images = None
        BaseImageGrabber.__init__(self)


    def Init(self):
        images = []
        for file in os.listdir(self.imageSourceFolder):
            if file.endswith(".jpg"):
                img_path = os.path.join(self.imageSourceFolder, file)
                print("Read image: {}".format(file))
                image = cv2.imread(img_path)
                images.append(image)

        self.images = images
        self.maxIterator = len(images)-1

    def GrabImage(self):
        currentImage = self.images[self.currentIterator]
        self.currentIterator = self.currentIterator + 1
        return currentImage

    def HasNext(self):
        """Check whether there are unprocessed images..
        """
        if self.currentIterator > self.maxIterator:
            return False
        else:
            return True

