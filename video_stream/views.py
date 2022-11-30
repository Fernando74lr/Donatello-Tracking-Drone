from django.http.response import StreamingHttpResponse
from video_stream.camera import Donatello
from django.shortcuts import render
from django.http import JsonResponse
from models import *
from utils.utils import *


# Create Donatello object
drone = Donatello()
# drone.generare_frame_ctrl()


# Index
def index(request):
    return render(request, 'video_stream/index.html')


# Generate normal frame
def generare_frame():
    # global drone
    # Infinite loop
    while True:
        # Get resized, flipped and encoded img
        # img = drone.get_frame()
        img = drone.generare_frame_ctrl()
        # Yield frame with its content-type
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')


# Get params
def get_params(request):
    height = drone.donatello.get_height()
    battery = drone.donatello.get_battery()
    temperature = drone.donatello.get_temperature()
    yaw = drone.donatello.get_yaw()
    flight_time = drone.donatello.get_flight_time()
    speed = drone.donatello.get_speed_x()

    return JsonResponse({"height": height, "battery": battery, "temperature": temperature, "yaw": yaw, "flight_time": flight_time, "speed": speed})


# Mode 0
def mode0(request):
    try:
        drone.optionMenu = 0
        return JsonResponse({"ok": True, "msg": 'Changed to mode0'})
    except:
        return JsonResponse({"ok": False, "msg": 'Error changing to mode0'})


# Mode 1
def mode1(request):
    try:
        drone.optionMenu = 1
        return JsonResponse({"ok": True, "msg": 'Changed to mode1'})
    except:
        return JsonResponse({"ok": False, "msg": 'Error changing to mode1'})


# Mode 2
def mode2(request):
    try:
        drone.optionMenu = 2
        return JsonResponse({"ok": True, "msg": 'Changed to mode2'})
    except:
        return JsonResponse({"ok": False, "msg": 'Error changing to mode2'})


# Start
def take_off(request):
    try:
        drone.donatello.takeoff()
        return JsonResponse({"ok": True})
    except:
        return JsonResponse({"ok": False})


# Land
def land(request):
    try:
        drone.donatello.land()
        drone.out.release()
        return JsonResponse({"ok": True})
    except:
        return JsonResponse({"ok": False})


# Move right
def move_right(request):
    try:
        drone.move_right()
        return JsonResponse({"ok": True})
    except:
        return JsonResponse({"ok": False})


# Move left
def move_left(request):
    try:
        drone.move_left()
        return JsonResponse({"ok": True})
    except:
        return JsonResponse({"ok": False})


# Move forward
def move_forward(request):
    try:
        drone.move_forward()
        return JsonResponse({"ok": True})
    except:
        return JsonResponse({"ok": False})


# Move backward
def move_backward(request):
    try:
        drone.move_backward()
        return JsonResponse({"ok": True})
    except:
        return JsonResponse({"ok": False})


# Move up
def move_up(request):
    try:
        drone.move_up()
        return JsonResponse({"ok": True})
    except:
        return JsonResponse({"ok": False})


# Move down
def move_down(request):
    try:
        drone.move_down()
        return JsonResponse({"ok": True})
    except:
        return JsonResponse({"ok": False})


# Video feed
def video_feed(request):
    return StreamingHttpResponse(generare_frame(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
