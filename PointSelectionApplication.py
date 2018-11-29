import cv2

imagePath = "./images/project_video_sample.png"
selectedPoints = []

def MouseClickHandler(event, x, y, flags, param):
    global selectedPoints
    if event == cv2.EVENT_LBUTTONDOWN:
        print("X, Y Coordinates of clicked point: {}, {}".format(x, y))
        selectedPoints.append((x,y))
    return

if __name__ == "__main__":
    #Read the specified image from disk
    image = cv2.imread(imagePath)

    # Register a window to OpenCV
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", MouseClickHandler)

    #Allow user to select points on image
    while True:
        cv2.imshow("Select Points", image)
        if cv2.waitKey(50) == ord('q'):
            cv2.destroyWindow("Select Points")
            break

    #Draw the points to the image
    for selectedPoint in selectedPoints:
        cv2.circle(image, selectedPoint, 1, (255,255,255))

    #Show the image with points
    cv2.imshow("Points", image)
    cv2.waitKey(0)