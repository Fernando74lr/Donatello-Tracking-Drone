from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # Admin url
    path('admin/', admin.site.urls),
    # Video stream urls
    path('', include('video_stream.urls'))
]
