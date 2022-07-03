from django.contrib import admin
from django.urls import path
from . import view
urlpatterns = [
    path('admin/',admin.site.urls),
    path('index/',view.index),
    path('uploadImage/',view.uploadImage,name='upload_iamge'),
    path('download/',view.downloadImage,name='download_iamge')
]

