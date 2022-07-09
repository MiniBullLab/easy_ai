from django.http import HttpResponse
from django.shortcuts import render
import os

from ConvertPic.utils.PngConvertTest import ConvertPng_Jpg


def index(request):
    context = {}
    context['hello'] = '你好'
    return render(request,'index.html',context)

def uploadImage(request):
    if  request.method == 'GET':
        return render(request,'index.html')
    elif request.method == 'POST':
        img = request.FILES.get("image",None)
        path='upload/'
        if not os.path.exists(path):
            os.mkdir(path)
        if not img:
            return HttpResponse("no files for upload")
        filename = img.name
        with open(path + filename,'wb') as f:
            for chunk in img.chunks():  # 分块写入文件
                f.write(chunk)
        url = path + filename
        ConvertPng_Jpg(url)
        #download_url ='http://127.0.0.1:8080' + '/' + path + filename
        # context = {}
        # context['path'] = path
        # context['filename'] = 'two.png'
        return render(request,'down.html')

def downloadImage(request):
    if  request.method == 'POST':
        pass
    try:
        path='upload/two.jpg'
        with open(path, 'rb') as f:
            file = f.read()

        response = HttpResponse(file)
        # file_name下载下来保存的文件名字
        file_name = path.split('/')[-1]
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment;filename={}'.format(file_name)
        return response
    except Exception as e:
        print('error:', e)
        return HttpResponse("error")
    # img_url = open(os.path.join("upload",img.name),"wb+")