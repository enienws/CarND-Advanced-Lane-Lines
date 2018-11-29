import abc

class BaseImageGrabber(metaclass=abc.ABCMeta):
    def __init__(self):
        self.currentIterator = 0
        self.maxIterator = 0

    @abc.abstractmethod
    def Init(self):
        """Initialize class variables and subsystems
        """

    @abc.abstractmethod
    def GrabImage(self):
        """Grab the image from image source and return the
        image to the client.
        """
    @abc.abstractmethod
    def HasNext(self):
        """Check whether there are unprocessed images..
        """