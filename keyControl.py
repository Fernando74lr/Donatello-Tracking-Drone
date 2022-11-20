from datetime import date, datetime
import imp
import keyPressModule as kp
import time
import cv2

kp.init()

# Keyboard Control
def getKeyboardInput(donatello, optionMenu, img):
    # left-right, forward-backward, up-down, yav
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    # KEYS

    if kp.getKey("RIGHT"): lr = speed
    elif kp.getKey("LEFT"): lr = -speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    # # TAKEOFF AND LAND
    if kp.getKey("e"): donatello.takeoff()
    if kp.getKey("q"): donatello.land()

    # BUTTONS

    if kp.getButtonPress("LEFT"):
        print('LEFT BUTTON')
        lr = speed
    if kp.getButtonPress("RIGHT"):
        print('RIGHT BUTTON')
        lr = -speed

    # if kp.getButtonPress("UP"): fb = speed
    # elif kp.getButtonPress("DOWN"): fb = -speed

    if kp.getButtonPress("UP"):
        print('UP BUTTON')
        ud = speed
    if kp.getButtonPress("DOWN"):
        print('DOWN BUTTON')
        ud = -speed

    if kp.getButtonPress("YAW-LEFT"):
        print("YAW-LEFT BUTTON")
        yv = speed
    if kp.getButtonPress("YAW-RIGHT"):
        print("YAW-RIGHT BUTTON")
        yv = -speed

    # TAKEOFF AND LAND
    if kp.getButtonPress("LAUNCH"):
        print('LAUNCH BUTTON')
        donatello.takeoff()
    if kp.getButtonPress("LAND"):
        print('LAND BUTTON')
        donatello.land()

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
    if kp.getKey("z") or kp.getButtonPress("PHOTO"):
        cv2.imwrite(f'Resources/Images/donatello_IM{datetime.timestamp(datetime.now())}.jpg', img)
        print("SS Take it")
        time.sleep(0.3)

    return lr, fb, ud, yv, optionMenu
    