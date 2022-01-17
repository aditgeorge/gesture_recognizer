from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_gen', views.video_gen, name='video_gen'),
    # path('sili', views.sili),
]
