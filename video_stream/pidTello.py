import cv2
import numpy as np
from simple_pid import PID

# Track using a simple PID
# v2 - Track using u and v axis from the camera
# Dron Frame -> x, y, z, yaw
# Camera values ->  u = current position of the center in the x axis of each frame
#                   v = current position of the center in the y axis of each frame
def track(donatello, pidForYaw, pidForZaxis, pidForXaxis, info, fbRange):
    # Get the face information [u, v][Area]
    area = info[1]
    u, v = info[0]
    fb = 0          # Forward velocity

    # PID for yaw and Z
    yawSpeed = int(-pidForYaw(u))
    updownSpeed = int(pidForZaxis(v))
    forwardSpeed = int(pidForXaxis(area))

    # Depth Algorithm
    # If its inside the interval -> NO MOVE
    # if area > fbRange[0] and area < fbRange[1]:
    #     forwardSpeed = 0

    # If no face track -> Reset parameters
    if u == 0: 
        yawSpeed = 0
        updownSpeed = 0
        forwardSpeed = 0

    print("u: ", u, " v: ", v, " area: ", area//1000)
    print("yS: ",yawSpeed, " udS: ", updownSpeed, " fbS: ", forwardSpeed, " fb: ", fb)

    # # Paralel control (Proposal):
    donatello.send_rc_control(0, forwardSpeed, updownSpeed, yawSpeed)
    # donatello.send_rc_control(0, 0, 0, yawSpeed)
    # donatello.send_rc_control(0, 0, updownSpeed, 0)
    # donatello.send_rc_control(0, forwardSpeed, 0, 0)
    
    # # Nested control (Proposal):
    # if(updownSpeed < 5 & updownSpeed > -5):
    #     if(yawSpeed < 10 & yawSpeed > -10):
    #         #donatello.send_rc_control(0, forwardSpeed, 0, 0)
    #         print("FORWARD UWU")
    #     else:
    #         donatello.send_rc_control(0, 0, 0, yawSpeed)
    # else:
    #     donatello.send_rc_control(0, 0, updownSpeed, 0)