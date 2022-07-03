from PIL import Image
import cv2 as cv
import os

def ConvertPng_Jpg(png_Path):
    #png_Path = png_Path.split('/')[-1]
    print(png_Path)
    img = cv.imread(png_Path,0)
    w,h = img.shape[::-1]
    fileInputPath = png_Path
    fileOutputPath = os.path.splitext(fileInputPath)[0] + ".jpg"
    img = Image.open(fileInputPath)
    img = img.resize((int(w/2),int(h/2)),Image.ANTIALIAS)
    try:
        if  len(img.split()) == 4:
            r,g,b,a = img.split()
            img = Image.merge("RGB",(r,g,b))
            img.convert('RGB').save(fileOutputPath,quality=70)
            #os.remove(png_Path)
        else:
            img.convert('RGB').save(fileOutputPath,quality=70)
            #os.remove(png_Path)
        return fileOutputPath
    except Exception as e:
        print("PNG转换JPG错误",e)


#ConvertPng_Jpg('../../upload/two.png')