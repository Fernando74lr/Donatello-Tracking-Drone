from django.urls import path
from video_stream import views

urlpatterns = [
    path('', views.index, name='home'),
    # Video feed
    path('video_feed', views.video_feed, name='video_feed'),
    # Movements
    path('right', views.move_right, name='move_right'),
    path('left', views.move_left, name='move_left'),
    path('forward', views.move_forward, name='move_forward'),
    path('backward', views.move_backward, name='move_backward'),
    path('up', views.move_up, name='move_up'),
    path('down', views.move_down, name='move_down'),
    # Take off
    path('take_off', views.take_off, name='take_off'),
    # Land
    path('land', views.land, name='land'),
    # Params
    path('get_params', views.get_params, name='get_params'),
    # Take photo
    path('take_photo', views.get_params, name='take_photo'),
    # Mode
    path('mode0', views.mode0, name='mode0'),
    path('mode1', views.mode1, name='mode1'),
    path('mode2', views.mode2, name='mode2'),
]
