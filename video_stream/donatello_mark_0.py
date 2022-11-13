from djitellopy import tello
import cv2
import time

# Create an tello object
donatello = tello.Tello()
donatello.connect()

# General information
print("Battery level: ", donatello.get_battery(), "%")
print("Temparature: ", donatello.get_temperature(), "~")

# Enables exchange of images
donatello.streamon()
time.sleep(1)

# Image size (width, height)
w, h = 360, 240

while True:
    # Get frames from Tello cam
    img = donatello.get_frame_read().frame
    img = cv2.resize(img, (w, h))

    # Displays
    cv2.imshow("Output", img)
    cv2.waitKey(1)
