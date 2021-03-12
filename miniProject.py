from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *
import math
import numpy as np
import cv2

###############
## 전역 변수
###############
inImage=outImage=tmpImage=[]
inH=inW=outH=outW=tmpH=tmpW=0
fileName=openFileName=''
paper=bitmap=image=None
RGB,RR,GG,BB=3,0,1,2
MAXSIZE=750
tmpFiles=[]
tmpIndex=0
inCvImage=outCvImage=tmpCvImage=None
selectedMenu,down_x, down_y, up_x, up_y = [-1]*5
boxLine = None

###############
## 공통 함수
###############
def malloc(row,col, init=0) :
    returnAry=[[[init for _ in range(col)] for _ in range(row)] for _ in range(RGB)]
    return returnAry

def open_Image():
    global inImage,outImage,tmpImage,inH,inW,outH,outW,tmpH,tmpW,fileName,openFileName
    global paper,bitmap,image,RGB,RR,GG,BB,inCvImage,outCvImage,tmpCvImage

    fileName = askopenfilename(parent=window,filetypes=(('Color File', '*.png;*.jpg;*.bmp;*.tif'), ('All File', '*.*')))

    inCvImage=cv2.imread(fileName)
    tmpCvImage = cv2.imread(fileName)

    inH,inW=inCvImage.shape[:2]
    tmpH=inH
    tmpW=inW

    inImage=malloc(inH,inW)
    tmpImage=malloc(tmpH,tmpW)

    for i in range(inH) :
        for k in range(inW) :
            inImage[BB][i][k]=tmpImage[BB][i][k]=inCvImage.item(i,k,RR)
            inImage[GG][i][k]=tmpImage[GG][i][k]=inCvImage.item(i,k,GG)
            inImage[RR][i][k]=tmpImage[RR][i][k]=inCvImage.item(i,k,BB)

    equal_Image()

def save_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if fileName==None :
        return

    saveCvImage=np.zeros((outH,outW,3),np.uint8)
    for i in range(outH) :
        for k in range(outW) :
            tup=tuple(([outImage[BB][i][k],outImage[GG][i][k],outImage[RR][i][k]]))
            saveCvImage[i][k]=tup

    saveFp = asksaveasfile(parent=window,mode='wb',defaultextension='.png',filetypes=(("Image Type", "*.png;*.jpg;*.bmp;*.tif"),("All File","*.*")))
    if(saveFp=='' or saveFp==None) :
        return
    cv2.imwrite(saveFp.name,saveCvImage)

def saveTempFile():
    pass
def restoreTempFile():
    pass
def display_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if bitmap != None :
        bitmap.destroy()

    window.geometry(str(outW)+'x'+str(outH))
    bitmap = Canvas(window,height=outH,width=outW)
    paper=PhotoImage(height=outH,width=outW)
    bitmap.create_image((outW/2,outH/2),image=paper,state='normal')

    rgbString=''
    for i in range(outH) :
        tmpString=''
        for k in range(outW) :
            rr=outImage[RR][i][k]
            gg=outImage[GG][i][k]
            bb=outImage[BB][i][k]
            tmpString+="#%02x%02x%02x "%(rr,gg,bb)
        rgbString+='{'+tmpString+'} '
    paper.put(rgbString)
    bitmap.pack()
    status.configure(text=str(outW)+'X'+str(outH)+'  '+fileName)

def CvToOutImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    outH,outW=outCvImage.shape[:2]
    outImage=malloc(outH,outW)

    for i in range(outH):
        for k in range(outW):
            if outCvImage.ndim==2:
                outImage[BB][i][k]=outCvImage.item(i,k)
                outImage[GG][i][k]=outCvImage.item(i,k)
                outImage[RR][i][k]=outCvImage.item(i,k)
            else:
                outImage[BB][i][k]=outCvImage.item(i,k,RR)
                outImage[GG][i][k]=outCvImage.item(i,k,GG)
                outImage[RR][i][k]=outCvImage.item(i,k,BB)


def OutImageToCv():
    pass

###############
## 범위지정 함수
###############
def LClick(event):
    global down_x, down_y, up_x, up_y

    down_x = event.x
    down_y = event.y

def Release(event):
    global down_x, down_y, up_x, up_y

    up_x = event.x
    up_y = event.y

    if down_x > up_x :
        down_x, up_x = up_x, down_x
    if down_y > up_y:
        down_y, up_y = up_y, down_y

    selected()
    bitmap.unbind("<Button-1>")
    bitmap.unbind("<B1-Motion>")
    bitmap.unbind("<Button-3>")
    bitmap.unbind("<ButtonRelease-1>")

def RClick(event):
    global down_x, down_y, up_x, up_y

    down_x=down_y=0
    up_x = outW-1
    up_y = outH-1

    selected()
    bitmap.unbind("<Button-1>")
    bitmap.unbind("<B1-Motion>")
    bitmap.unbind("<Button-3>")
    bitmap.unbind("<ButtonRelease-1>")

def Move(event):
    global down_x, down_y, up_x, up_y, boxLine

    if down_x<0:
        return
    up_x=event.x
    up_y=event.y

    if not boxLine:
        pass
    else:
        bitmap.delete(boxLine)
    boxLine=bitmap.create_rectangle(down_x,down_y,up_x,up_y,fill=None)

def selected():
    global selectedMenu

    if selectedMenu == 1:
        __bright_Image()
    elif selectedMenu == 2:
        __bw_Image()
    elif selectedMenu == 3:
        __color_Reverse_Image()
    elif selectedMenu == 4:
        __mosaic1_Image()
    elif selectedMenu == 5:
        __mosaic2_Image()

###############
## 영상처리 함수
###############
def equal_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None :
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH,outW)

    for rgb in range(RGB) :
        for i in range(outH) :
            for k in range(outW) :
                outImage[rgb][i][k] = tmpImage[rgb][i][k]

    display_Image()
    saveTempFile()
    OutImageToCv()

def original_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH = inH
    outW = tmpW = inW
    outImage = malloc(outH, outW)

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpImage[rgb][i][k] = inImage[rgb][i][k]
                outImage[rgb][i][k] = inImage[rgb][i][k]

    display_Image()
    saveTempFile()
    OutImageToCv()

def bright_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu,down_x, down_y, up_x, up_y, boxLine
    selectedMenu=1

    if inImage == None:
        return
    messagebox.showinfo('범위 지정','좌 클릭 : 선택 범위 / 우 클릭 : 전체 범위')

    bitmap.bind("<Button-1>", LClick)
    bitmap.bind("<B1-Motion>", Move)
    bitmap.bind("<Button-3>", RClick)
    bitmap.bind("<ButtonRelease-1>", Release)

def __bright_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu, down_x, down_y, up_x, up_y, boxLine

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    bright = askinteger("밝기 입력", "밝기를 입력하세요. (+:밝게/-:어둡게)", minvalue=-255, maxvalue=255)
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = tmpImage[rgb][i][k]
                if (down_x <= k <= up_x) and (down_y <= i <= up_y):
                    if ((tmpImage[rgb][i][k] + bright) > 255):
                        outImage[rgb][i][k] = 255
                    elif ((tmpImage[rgb][i][k] + bright) < 0):
                        outImage[rgb][i][k] = 0
                    else:
                        outImage[rgb][i][k] = tmpImage[rgb][i][k] + bright
                else:
                    outImage[rgb][i][k] = tmpImage[rgb][i][k]

    display_Image()
    saveTempFile()
    OutImageToCv()

def bw_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu, down_x, down_y, up_x, up_y, boxLine
    selectedMenu = 2

    if inImage == None:
        return
    messagebox.showinfo('범위 지정', '좌 클릭 : 선택 범위 / 우 클릭 : 전체 범위')

    bitmap.bind("<Button-1>", LClick)
    bitmap.bind("<B1-Motion>", Move)
    bitmap.bind("<Button-3>", RClick)
    bitmap.bind("<ButtonRelease-1>", Release)

def __bw_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu, down_x, down_y, up_x, up_y, boxLine

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    sum = 0
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                sum += tmpImage[rgb][i][k]
    avg = sum / (outH * outW * RGB)

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                rgb_avg = (tmpImage[RR][i][k] + tmpImage[GG][i][k] + tmpImage[BB][i][k]) / RGB
                if (down_x <= k <= up_x) and (down_y <= i <= up_y):
                    if rgb_avg < avg :
                        outImage[rgb][i][k] = 0
                    else:
                        outImage[rgb][i][k] = 255
                else:
                    outImage[rgb][i][k] = tmpImage[rgb][i][k]

    display_Image()
    saveTempFile()
    OutImageToCv()

def color_Reverse_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu, down_x, down_y, up_x, up_y, boxLine
    selectedMenu = 3

    if inImage == None:
        return
    messagebox.showinfo('범위 지정','좌 클릭 : 선택 범위 / 우 클릭 : 전체 범위')

    bitmap.bind("<Button-1>", LClick)
    bitmap.bind("<B1-Motion>", Move)
    bitmap.bind("<Button-3>", RClick)
    bitmap.bind("<ButtonRelease-1>", Release)

def __color_Reverse_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu, down_x, down_y, up_x, up_y, boxLine

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if (down_x <= k <= up_x) and (down_y <= i <= up_y):
                    outImage[rgb][i][k] = 255 - tmpImage[rgb][i][k]
                else:
                    outImage[rgb][i][k] = tmpImage[rgb][i][k]

    display_Image()
    saveTempFile()
    OutImageToCv()

def rl_Reverse_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = tmpImage[rgb][i][outW - k - 1]

    display_Image()
    saveTempFile()
    OutImageToCv()

def ud_Reverse_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = tmpImage[rgb][outH - i - 1][k]

    display_Image()
    saveTempFile()
    OutImageToCv()

def scaleUp_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH*2
    outW = tmpW*2
    outImage = malloc(outH, outW)

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = tmpImage[rgb][i // 2][k // 2]

    display_Image()
    saveTempFile()
    OutImageToCv()

def scaleDown_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH // 2
    outW = tmpW // 2
    outImage = malloc(outH, outW)

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = tmpImage[rgb][i * 2][k * 2]

    display_Image()
    saveTempFile()
    OutImageToCv()

def move_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    y = askinteger("이동 방향 및 거리 입력", "이동 방향과 거리를 입력하세요. (+:우/-:좌)", minvalue=-outW, maxvalue=outW)
    x = askinteger("이동 방향 및 거리 입력", "이동 방향과 거리를 입력하세요. (+:하/-:상)", minvalue=-outH, maxvalue=outH)

    for rgb in range(RGB):
        for i in range(outH):
            if (i + y) < 0 or (i + y) >= outH:
                continue
            for k in range(outW):
                if (k + x) < 0 or (k + x) >= outW:
                    continue;
                outImage[rgb][i + y][k + x] = tmpImage[rgb][i][k]

    display_Image()
    saveTempFile()
    OutImageToCv()

def rotate_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    deg = askinteger("회전 각도 입력", "회전할 각도를 입력하세요. (시계방향)", minvalue=0, maxvalue=360)
    PI = 3.141592
    rad = -deg * PI / 180.0

    outH = round(abs(tmpW * math.sin(rad)) + abs(tmpH * math.cos(rad)))+10
    outW = round(abs(tmpH * math.sin(rad)) + abs(tmpW * math.cos(rad)))+10
    if outH < tmpH:
        outH = tmpH
    if outW < tmpW:
        outW = tmpW
    outImage = malloc(round(outH), round(outW))
    rot_Image = malloc(round(outH), round(outW))

    new_x, new_y=0,0
    center_x,center_y = outW / 2, outH / 2

    for rgb in range(RGB):
        a=0
        for i in range(round(center_y - (tmpH / 2)),round(center_y + (tmpH / 2)),1):
            b=0
            for k in range(round(center_x - (tmpW / 2)),round(center_x + (tmpW / 2)),1):
                rot_Image[rgb][i][k] = tmpImage[rgb][a][b]
                b+=1
            a+=1

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                new_x = round((i - center_y) * math.sin(rad) + (k - center_x) * math.cos(rad) + center_x)
                new_y = round((i - center_y) * math.cos(rad) - (k - center_x) * math.sin(rad) + center_y)
                if new_y < 0 or new_y >= outH:
                    outImage[rgb][i][k] = 0
                elif new_x < 0 or new_x >= outW:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = rot_Image[rgb][new_y][new_x]

    display_Image()
    saveTempFile()
    OutImageToCv()

def mosaic1_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu, down_x, down_y, up_x, up_y, boxLine
    selectedMenu = 4

    if inImage == None:
        return
    messagebox.showinfo('범위 지정', '좌 클릭 : 선택 범위 / 우 클릭 : 전체 범위')

    bitmap.bind("<Button-1>", LClick)
    bitmap.bind("<B1-Motion>", Move)
    bitmap.bind("<Button-3>", RClick)
    bitmap.bind("<ButtonRelease-1>", Release)

def __mosaic1_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu, down_x, down_y, up_x, up_y, boxLine

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k]=tmpImage[rgb][i][k]

    mosaic,mi,mk = 8,0,0
    for rgb in range(RGB):
        for i in range(0, outH, mosaic):
            for k in range(0, outW, mosaic):
                if (down_x <= k <= up_x) and (down_y <= i <= up_y):
                    if ((i + mosaic) > outH) or ((k + mosaic) > outW):
                        for a in range(outH % mosaic):
                            for b in range(outW % mosaic):
                                outImage[rgb][i + a][k + b] = tmpImage[rgb][i + a][k + b]
                    else:
                        for a in range(mosaic):
                            for b in range(mosaic):
                                outImage[rgb][i + a][k + b] = tmpImage[rgb][i + (mosaic // 2)][k + (mosaic // 2)]

    for rgb in range(RGB):
        for i in range(outH-mosaic,outH,1):
            for k in range(outW-mosaic,outW,1):
                outImage[rgb][i][k] = tmpImage[rgb][i][k]

    display_Image()
    saveTempFile()
    OutImageToCv()

def mosaic2_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu, down_x, down_y, up_x, up_y, boxLine
    selectedMenu = 5

    if inImage == None:
        return
    messagebox.showinfo('범위 지정', '좌 클릭 : 선택 범위 / 우 클릭 : 전체 범위')

    bitmap.bind("<Button-1>", LClick)
    bitmap.bind("<B1-Motion>", Move)
    bitmap.bind("<Button-3>", RClick)
    bitmap.bind("<ButtonRelease-1>", Release)

def __mosaic2_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage
    global selectedMenu, down_x, down_y, up_x, up_y, boxLine

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = tmpImage[rgb][i][k]

    mosaic, mi, mk = 16, 0, 0
    for rgb in range(RGB):
        for i in range(0, outH, mosaic):
            for k in range(0, outW, mosaic):
                if (down_x <= k <= up_x) and (down_y <= i <= up_y):
                    if ((i + mosaic) > outH) or ((k + mosaic) > outW):
                        for a in range(outH % mosaic):
                            for b in range(outW % mosaic):
                                outImage[rgb][i + a][k + b] = tmpImage[rgb][i + a][k + b]
                    else:
                        for a in range(mosaic):
                            for b in range(mosaic):
                                outImage[rgb][i + a][k + b] = tmpImage[rgb][i + (mosaic // 2)][k + (mosaic // 2)]

    for rgb in range(RGB):
        for i in range(outH - mosaic, outH, 1):
            for k in range(outW - mosaic, outW, 1):
                outImage[rgb][i][k] = tmpImage[rgb][i][k]

    display_Image()
    saveTempFile()
    OutImageToCv()

def emboss_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    MSIZE=3
    mask = [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    tmp_input=malloc(tmpH+2,tmpW+2)
    tmp_output=malloc(tmpH,tmpW)


    for rgb in range(RGB):
        for i in range(outH+2):
            for k in range(outW+2):
                tmp_input[rgb][i][k] = 127

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmp_input[rgb][i+1][k+1] = tmpImage[rgb][i][k]

    sum=0.0
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                for a in range(MSIZE):
                    for b in range(MSIZE):
                        sum+=tmp_input[rgb][i + a][k + b]*mask[a][b]
                tmp_output[rgb][i][k]=sum+127
                sum=0.0

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmp_output[rgb][i][k]>255:
                    tmp_output[rgb][i][k]=255
                elif tmp_output[rgb][i][k]<0:
                    tmp_output[rgb][i][k]=0
                outImage[rgb][i][k]=round(tmp_output[rgb][i][k])

    display_Image()
    saveTempFile()
    OutImageToCv()

def blurr_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    MSIZE = 3
    mask = [[ 1 / 9.0, 1 / 9.0, 1 / 9.0 ], [ 1 / 9.0, 1 / 9.0, 1 / 9.0 ], [ 1 / 9.0, 1 / 9.0, 1 / 9.0 ]]
    tmp_input = malloc(tmpH + 2, tmpW + 2)
    tmp_output = malloc(tmpH, tmpW)

    for rgb in range(RGB):
        for i in range(outH + 2):
            for k in range(outW + 2):
                tmp_input[rgb][i][k] = 127

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmp_input[rgb][i + 1][k + 1] = tmpImage[rgb][i][k]

    sum = 0.0
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                for a in range(MSIZE):
                    for b in range(MSIZE):
                        sum += tmp_input[rgb][i + a][k + b] * mask[a][b]
                tmp_output[rgb][i][k] = sum
                sum = 0.0

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmp_output[rgb][i][k] > 255:
                    tmp_output[rgb][i][k] = 255
                elif tmp_output[rgb][i][k] < 0:
                    tmp_output[rgb][i][k] = 0
                outImage[rgb][i][k] = round(tmp_output[rgb][i][k])

    display_Image()
    saveTempFile()
    OutImageToCv()

def edge_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    mask_x=[[ -1, 0, 1 ], [ -2, 0, 2 ], [ -1, 0, 1 ]]
    mask_y=[[ -1, -2, -1 ], [ 0, 0, 0 ], [ 1, 2, 1 ]]

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k]=0

    for rgb in range(RGB):
        for i in range(1,outH-1,1):
            for k in range(1,outW-1,1):
                sum_x,sum_y = 0,0
                for a in range(3):
                    for b in range(3):
                        sum_x += mask_x[a][b] * tmpImage[rgb][i + a - 1][k + b - 1]
                        sum_y += mask_y[a][b] * tmpImage[rgb][i + a - 1][k + b - 1]
                mag = math.sqrt(sum_x * sum_x + sum_y * sum_y)
                if mag>255:
                    mag=255
                elif mag<0:
                    mag=0
                outImage[rgb][i][k]=round(mag)

    display_Image()
    saveTempFile()
    OutImageToCv()

def stretch_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    min_num = max_num = tmpImage[0][0][0]
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if min_num>tmpImage[rgb][i][k]:
                    min_num = tmpImage[rgb][i][k]
                elif max_num<tmpImage[rgb][i][k]:
                    max_num = tmpImage[rgb][i][k]

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = round((tmpImage[rgb][i][k] - min_num) / (max_num - min_num) * 255)

    display_Image()
    saveTempFile()
    OutImageToCv()

def endin_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    min_num = max_num = tmpImage[0][0][0]
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if min_num > tmpImage[rgb][i][k]:
                    min_num = tmpImage[rgb][i][k]
                elif max_num < tmpImage[rgb][i][k]:
                    max_num = tmpImage[rgb][i][k]
    min_num+=50
    max_num-=50

    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                value = (tmpImage[rgb][i][k] - min_num) / (max_num - min_num) * 255
                if value>255:
                    value=255
                elif value<0:
                    value=0
                outImage[rgb][i][k] = round(value)

    display_Image()
    saveTempFile()
    OutImageToCv()

def  hisequal_Image():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outH = tmpH
    outW = tmpW
    outImage = malloc(outH, outW)

    for rgb in range(RGB):
        sum=0
        hist=sum_hist=normal_hist=[0 for i in range(256)]

        for i in range(tmpH):
            for k in range(tmpW):
                hist[tmpImage[rgb][i][k]]+=1
        for i in range(256):
            sum+=hist[i]
            sum_hist[i]=sum
        for i in range(256):
            normal_hist[i] = sum_hist[i] / (tmpH * tmpW) *255.0
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k]=round(normal_hist[tmpImage[rgb][i][k]])

    display_Image()
    saveTempFile()
    OutImageToCv()

###############
## OpenCV 함수
###############
def original_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outCvImage=inCvImage[:].copy()
    tmpCvImage=inCvImage[:].copy()

    CvToOutImage()
    display_Image()

def cut_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    start_X = askinteger("자르기 입력","자르기 시작 할 X지점을 입력하세요.", minvalue=0, maxvalue=outW)
    width = askinteger("자르기 입력","자를 이미지의 폭을 입력하세요.", minvalue=0, maxvalue=outW)
    start_Y = askinteger("자르기 입력","자르기 시작 할 Y지점을 입력하세요.", minvalue=0, maxvalue=outH)
    height = askinteger("자르기 입력","자를 이미지의 높이를 입력하세요.", minvalue=0, maxvalue=outH)
    if (width == 0 or height == 0):
        return

    outCvImage = tmpCvImage[start_Y:height, start_X:width].copy()

    CvToOutImage()
    display_Image()

def bw_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outCvImage=cv2.cvtColor(tmpCvImage,cv2.COLOR_RGB2GRAY)
    outCvImage=cv2.adaptiveThreshold(outCvImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,5)

    CvToOutImage()
    display_Image()

def grayscale_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outCvImage=cv2.cvtColor(tmpCvImage,cv2.COLOR_BGR2GRAY)

    CvToOutImage()
    display_Image()

def emboss_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    mask=np.zeros((3,3),np.float32)
    mask[0][0]=-1.0
    mask[2][2]=1.0


    outCvImage=cv2.filter2D(tmpCvImage,-1,mask)
    outCvImage+=127

    CvToOutImage()
    display_Image()

def cartoon_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outCvImage=cv2.cvtColor(tmpCvImage,cv2.COLOR_BGR2GRAY)
    outCvImage=cv2.medianBlur(outCvImage,7)
    edges=cv2.Laplacian(outCvImage,cv2.CV_8U,ksize=5)
    ret,mask=cv2.threshold(edges,100,255,cv2.THRESH_BINARY_INV)
    outCvImage=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

    CvToOutImage()
    display_Image()

def dilate_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    outCvImage = cv2.dilate(tmpCvImage, kernel, anchor=(-1, -1), iterations=5)

    CvToOutImage()
    display_Image()

def erode_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    outCvImage = cv2.erode(tmpCvImage, kernel, anchor=(-1, -1), iterations=5)

    CvToOutImage()
    display_Image()

def edge_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outCvImage=tmpCvImage[:].copy()
    bin = tmpCvImage[:].copy()

    bin=cv2.cvtColor(tmpCvImage, cv2.COLOR_BGR2GRAY)
    ret,bin=cv2.threshold(bin, 127, 255, cv2.THRESH_BINARY)
    bin = cv2.bitwise_not(bin)

    contour, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contour)):
        cv2.drawContours(outCvImage, [contour[i]], 0, (255, 0, 0), 3)

    CvToOutImage()
    display_Image()

def corner_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outCvImage=tmpCvImage[:].copy()

    gray = cv2.cvtColor(tmpCvImage, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)

    for i in corners:
        cv2.circle(outCvImage, tuple(i[0]), 1, (255, 0, 0), 7)

    CvToOutImage()
    display_Image()

def circle_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outCvImage=tmpCvImage[:].copy()

    gray = cv2.cvtColor(tmpCvImage, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=250, param2=10, minRadius=0, maxRadius=100)

    for i in circles[0]:
        cv2.circle(outCvImage, (i[0], i[1]), round(i[2]), (255, 0, 0), 4)

    CvToOutImage()
    display_Image()

def color_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    low_Color = askinteger("색상 입력","검출 범위의 시작 색상을 입력하세요.", minvalue=0, maxvalue=360)
    up_Color = askinteger("색상 입력","검출 범위의 끝 색상을 입력하세요.", minvalue=0, maxvalue=360)

    tmpCvImage = cv2.cvtColor(tmpCvImage, cv2.COLOR_RGB2HSV)
    mv=cv2.split(tmpCvImage)
    tmpCvImage = cv2.cvtColor(tmpCvImage, cv2.COLOR_HSV2RGB)
    mask=cv2.inRange(mv[0],low_Color,up_Color)
    mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    outCvImage=cv2.bitwise_and(tmpCvImage,mask)

    CvToOutImage()
    display_Image()

def center_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outCvImage=tmpCvImage[:].copy()

    gray = cv2.cvtColor(tmpCvImage, cv2.COLOR_RGB2GRAY)
    ret, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        mmt = cv2.moments(i)
        cx = int(mmt['m10'] / mmt['m00'])
        cy = int(mmt['m01'] / mmt['m00'])
        cv2.circle(outCvImage, (cx, cy), 1, (255, 0, 0), 7)

    CvToOutImage()
    display_Image()

def quadrangle_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    outCvImage=tmpCvImage[:].copy()

    gray = cv2.cvtColor(tmpCvImage, cv2.COLOR_RGB2GRAY)
    ret, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(outCvImage, [box], 0, (255, 0, 0), 3)

    CvToOutImage()
    display_Image()

def face_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    face_cascade=cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    gray=cv2.cvtColor(tmpCvImage,cv2.COLOR_RGB2GRAY)
    face_rects=face_cascade.detectMultiScale(gray,1.1,5)

    outCvImage=tmpCvImage[:].copy()
    for (x,y,w,h) in face_rects :
        cv2.rectangle(outCvImage,(x,y),(x+w,y+h),(255,0,0),2)

    CvToOutImage()
    display_Image()

def equalArea_CvImage():
    global inImage, outImage, tmpImage, inH, inW, outH, outW, tmpH, tmpW, fileName, openFileName
    global paper, bitmap, image, RGB, RR, GG, BB, inCvImage, outCvImage, tmpCvImage

    if inImage == None:
        return

    messagebox.showinfo('템플릿 파일 열기','원본 이미지와 비교할 템플릿 이미지를 선택해주세요.')
    templitFileName = askopenfilename(parent=window,filetypes=(('Color File', '*.png;*.jpg;*.bmp;*.tif'), ('All File', '*.*')))
    templitCvImage = cv2.imread(templitFileName,cv2.IMREAD_GRAYSCALE)
    gray=cv2.cvtColor(tmpCvImage,cv2.COLOR_RGB2GRAY)
    outCvImage=tmpCvImage[:].copy()

    result = cv2.matchTemplate(gray, templitCvImage, cv2.TM_SQDIFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    x, y = minLoc
    h, w = templitCvImage.shape

    outCvImage = cv2.rectangle(outCvImage, (x, y), (x + w, y + h), (255, 0, 0), 3)

    CvToOutImage()
    display_Image()

###############
## 메인 함수
###############
window=Tk()
window.title("영상처리 프로그램")
window.geometry('500x400')
window.resizable(width=False,height=False)

status=Label(window,text='이미지 정보 :',bd=1,relief=SUNKEN,anchor=W)
status.pack(side=BOTTOM,fill=X)

mainMenu=Menu(window)
window.config(menu=mainMenu)

fileMenu=Menu(mainMenu)
mainMenu.add_cascade(label='파일',menu=fileMenu)
fileMenu.add_command(label='열기',command=open_Image)
fileMenu.add_command(label='저장',command=save_Image)
fileMenu.add_separator()
fileMenu.add_command(label='종료',command=None)

imageMenu=Menu(mainMenu)
mainMenu.add_cascade(label='영상처리',menu=imageMenu)
imageMenu.add_command(label='원본',command=original_Image)
imageMenu.add_command(label='밝기 조절',command=bright_Image)
scaleMenu=Menu(imageMenu)
imageMenu.add_cascade(label='크기 조절',menu=scaleMenu)
scaleMenu.add_command(label='2배 확대',command=scaleUp_Image)
scaleMenu.add_command(label='2배 축소',command=scaleDown_Image)
imageMenu.add_command(label='흑백',command=bw_Image)
reverseMenu=Menu(imageMenu)
imageMenu.add_cascade(label='반전',menu=reverseMenu)
reverseMenu.add_command(label='색상 반전',command=color_Reverse_Image)
reverseMenu.add_command(label='좌/우 반전',command=rl_Reverse_Image)
reverseMenu.add_command(label='상/하 반전',command=ud_Reverse_Image)
imageMenu.add_command(label='이동',command=move_Image)
imageMenu.add_command(label='회전',command=rotate_Image)
mosaicMenu=Menu(imageMenu)
imageMenu.add_cascade(label='모자이크',menu=mosaicMenu)
mosaicMenu.add_command(label='약하게',command=mosaic1_Image)
mosaicMenu.add_command(label='강하게',command=mosaic2_Image)
imageMenu.add_command(label='경계선 검출',command=edge_Image)
imageMenu.add_command(label='엠보싱',command=emboss_Image)
imageMenu.add_command(label='블러링',command=blurr_Image)
stretchMenu=Menu(imageMenu)
imageMenu.add_cascade(label='선명하게',menu=stretchMenu)
stretchMenu.add_command(label='스트레칭',command=stretch_Image)
stretchMenu.add_command(label='엔드-인 탐색',command=endin_Image)
imageMenu.add_command(label='평활화',command=hisequal_Image)

cvMenu=Menu(mainMenu)
mainMenu.add_cascade(label='OpenCV',menu=cvMenu)
cvMenu.add_command(label='원본',command=original_CvImage)
cvMenu.add_command(label='자르기',command=cut_CvImage)
cvMenu.add_command(label='그레이 스케일',command=grayscale_CvImage)
cvMenu.add_command(label='적응형 이진화',command=bw_CvImage)
cvMenu.add_command(label='엠보싱',command=emboss_CvImage)
cvMenu.add_command(label='카툰 효과',command=cartoon_CvImage)
morpMenu=Menu(cvMenu)
cvMenu.add_cascade(label='모포로지',menu=morpMenu)
morpMenu.add_command(label='팽창',command=dilate_CvImage)
morpMenu.add_command(label='침식',command=erode_CvImage)
cvMenu.add_command(label='윤곽선 검출',command=edge_CvImage)
cvMenu.add_command(label='코너 검출',command=corner_CvImage)
cvMenu.add_command(label='원 검출',command=circle_CvImage)
cvMenu.add_command(label='색상 추출',command=color_CvImage)
cvMenu.add_command(label='중심점 추출',command=center_CvImage)
cvMenu.add_command(label='경계 사각형 추출',command=quadrangle_CvImage)
cvMenu.add_command(label='얼굴인식',command=face_CvImage)
cvMenu.add_command(label='동일 영역 검출',command=equalArea_CvImage)

window.mainloop()
