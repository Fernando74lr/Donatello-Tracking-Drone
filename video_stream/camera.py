from djitellopy import tello
import cv2


class VideoCamera(object):
    def __init__(self):
        # Initialize camera 0
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # Release camera video
        self.video.release()

    def get_frame(self):
        _, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        frame_flip = cv2.flip(image, 1)
        _, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()


class Donatello(object):
    def __init__(self):
        # Donatello obj
        self.donatello = tello.Tello()
        # Image size (width, height)
        self.w, self.h = 360, 240
        try:
            self.donatello.connect()
            self.donatello.streamon()
            print(f'\nConnected succesfully!\n')
        except Exception as error:
            print(f'\nError trying to connect:\n{error}\n')

    def __del__(self):
        # Release camera video
        self.donatello.land()

    def get_frame(self):
        # Get frame from camera
        frame = self.donatello.get_frame_read().frame
        # Resize with defined width and height
        frame = cv2.resize(frame, (self.w, self.h))
        # Flip
        frame_flip = cv2.flip(frame, 1)
        # Encode image and convert it to 'jpg' format
        _, img = cv2.imencode('.jpg', frame_flip)
        return img.tobytes()

    # Move right
    def move_right(self):
        self.donatello.move_right(35)

    # Move left
    def move_left(self):
        self.donatello.move_left(35)

    # Move forward
    def move_forward(self):
        self.donatello.move_forward(35)

    # Move backward
    def move_backward(self):
        self.donatello.move_back(35)

    # Move up
    def move_up(self):
        self.donatello.move_up(35)

    # Move down
    def move_down(self):
        self.donatello.move_down(35)
