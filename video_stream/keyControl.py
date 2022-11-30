from datetime import date, datetime
import imp
import video_stream.keyPressModule as kp
import time
import cv2

# init keyboard
kp.init()

# Keyboard Control
def getKeyboardInput(donatello, optionMenu, img):
    # left-right, forward-backward, up-down, yav
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if kp.getKey("RIGHT"): lr = speed
    elif kp.getKey("LEFT"): lr = -speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    # TAKEOFF AND LAND
    if kp.getKey("e"): donatello.takeoff()
    if kp.getKey("q"): donatello.land()

    # CHANGE MODE
    if kp.getKey("m"):
        if optionMenu == 0 :
            optionMenu = 1
        elif optionMenu == 1 :
            optionMenu = 2
        elif optionMenu == 2 :
            optionMenu = 0
        else :
            optionMenu == 0
        time.sleep(0.3)
        print("Mode changed to ", optionMenu)

    # GENERAL INFORMATION
    if kp.getKey("b"): print("Battery level: ", donatello.get_battery(), "%")
    if kp.getKey("t"): print("Temparature: ", donatello.get_temperature(), "~")

    # IMAGE CAPTURE
    if kp.getKey("z"):
        cv2.imwrite(f'Resources/Images/donatello_IM{datetime.timestamp(datetime.now())}.jpg', img)
        print("SS Take it")
        time.sleep(0.3)

    return lr, fb, ud, yv, optionMenu
    