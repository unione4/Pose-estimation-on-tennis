#射影変換を行い選手の位置情報を取得する(動画上と実際のコート上の選手の位置は異なるため)

import torch 
import cv2 
import numpy as np
import queue
from collections import deque
import math 
import os

sequence = 0
frame_num = 0

#here
file_num = 719
casted_num = str(file_num) 

#データ取得ファイル
course = "pull"
hand = "forehand"

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
video = cv2.VideoCapture("C:/impact_tennis/cap/" + course + "/" + hand + "/"  + casted_num + ".mp4")

data_path = "C:/impact_tennis/forehand/MP_Data/"+ course +  "/" + casted_num + "/"
DATA_PATH = "C:/impact_tennis/forehand/MP_last_Data/" + course + "/"

os.makedirs(os.path.join(DATA_PATH, casted_num))  #ファイル作成


#コート上の点
t_point = np.float32([[145,185],[475,185],[145,905],[475,905]])

#yoloのモデルクラス
model.classes=[0]

#キュー
d = deque([],5)

#動画の1フレーム目を取得
while video.isOpened():
    count=0
    ret, frame = video.read()
    if not ret:
        break
    if count == 0:
        img = frame
    count+=1
    break
    
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

out = cv2.VideoWriter("yolo_tracking_court.mp4", fourcc, 30, (1920,1080))

total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#コートの4隅の座標
four_angle = []

#マウスポイントで座標を取得する
def click_pos(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        img2 = np.copy(img)
        cv2.circle(img2,center=(x,y),radius=5,color=255,thickness=-1)
        pos_str='(x,y)=('+str(x)+','+str(y)+')'
        cv2.putText(img2,pos_str,(x+10, y+10),cv2.FONT_HERSHEY_PLAIN,2,255,2,cv2.LINE_AA)
        cv2.imshow('window', img2)
        print(x,y)

        four_angle.append((x,y))


cv2.imshow('window', img)
cv2.setMouseCallback('window', click_pos)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(four_angle)

#コートの4隅
point1 = np.array(four_angle)
point1 = np.float32(point1)

#射影変換を行う
def homography(img,point1,point2,ps):
    img2 = img.copy()
    M = cv2.getPerspectiveTransform(point1,point2)
    P = np.dot(M,ps)
    P = P/P[2]
    #print(np.dot(M,arr))
    #print(P)
    
    #選手として検出しない範囲の指定
    if (ps[1] > 1050) or (P[0]<145 and 275<P[1]<815) or (P[0]>475 and 250<P[1]<700) or (P[0]>550) or (P[1]>1080)or(P[1]<-50) or (P[0]<100):
        return 
    else:
        return (int(P[0]),int(P[1]))

#選手の位置座標を取得
def loop_draw_position(keypoints_with_scores):
    position = []
    count = 0
    avg_pos =[]

    img2 = cv2.imread("blank4.png")
    for num in range(len(keypoints_with_scores)):
        a = homography(img2,point1, t_point,[int(int(keypoints_with_scores[num][0]+keypoints_with_scores[num][2])/2),int(keypoints_with_scores[num][3]),1])
        if a != None:
            position.append(a)
        if homography(img2,point1, t_point,[int((keypoints_with_scores[num][0]+keypoints_with_scores[num][2])/2),int(keypoints_with_scores[num][3]),1]) ==1:
            count+=1
    
    #print('pos'+str(position))
    position = sorted(position,key=lambda x:x[1])
    #print('true_pos'+str(position))
    position = position[-2:]
    #print('f_pos'+str(position))

    d.append(position)
    print(d)

    #選手が2人検出されなかった場合は前のフレームから探索
    if frame_num > 4 and len(d[4]) < 2 :
        print(print(len(d[4])))
        #前のフレームとの比較で検出に失敗した選手を探索
        if distance(d[3][0][0],d[3][0][1],d[4][0][0],d[4][0][1]) < 100:
            d[4].append(d[3][1])
        else:
            d[4].append(d[3][0])
        
        d[4] = sorted(d[4],key=lambda x:x[1])


    #位置平均を出力する
    if (frame_num<5):
        avg_pos.append(d[-1][0])
        avg_pos.append(d[-1][1])    


    if frame_num > 4:
        for pos in d[4]:
            sum_num = pos
            length = 1
            for i in range(4):
                close_pos = ()
                min_num = 100
                for j in range(2):
                    a = distance(pos[0],pos[1],d[i][j][0],d[i][j][1])
                    if a < min_num:
                        min_num = a
                        close_pos = (d[i][j][0],d[i][j][1])
                if min_num != 100:
                    print(close_pos)
                    print(sum_num[0]+close_pos[0],sum_num[1]+close_pos[1])
                    sum_num = (sum_num[0]+close_pos[0],sum_num[1]+close_pos[1])
                    length+=1
            avg_pos.append((sum_num[0]/length,sum_num[1]/length))
    print('avg_pos=' + str(avg_pos))

    
    for tmp in avg_pos:
        cv2.circle(img2,(int(tmp[0]),int(tmp[1])),20,(0,0,0),thickness=3)
    
    #選手の位置を正規化
    new_pos = []
    for n in avg_pos:
        new_pos.append(n[0]/475)
        new_pos.append(n[1]/905)

    print("new_pos is " +  str(new_pos))

    cv2.imshow("ps_poos",img2)
    out.write(img2)

    #位置情報をファイルに書き込み
    if total_frame - 32 < frame_num:
        file_name = np.load(data_path + str(sequence)+'.npy')
        print("file_name is " + str(file_name))
        target = np.append(file_name,new_pos)
        file_name = target
        if len(target)!=96:
            print("ELEMENT ERROR")
            quit()   
        print(target) 
        npy_path = os.path.join(DATA_PATH, casted_num ,str(sequence))
        np.save(npy_path, target)

#2点間の距離を計算
def distance(x1,y1,x2,y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d


while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    print('a')
    result = model(frame)
    result.render()         #self.imgsに結果を描画した画像を格納する
    cv2.imshow("test",result.imgs[0]) 
    loop_draw_position(result.xyxy[0])#テニスコート上に描画
    if total_frame -32 < frame_num:
        sequence +=1

    frame_num+=1
           
    #result.display(show=True)
    #w_video.write(result.save) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
