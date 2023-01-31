#BlazePoseを用いた姿勢推定
import torch 
import cv2 
import numpy as np
import os 
from matplotlib import pyplot as plt
import time 
import mediapipe as mp

data_num = 719

casted_num = str(data_num)

course = "pull"
hand = "forehand"

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
video = cv2.VideoCapture("C:/impact_tennis/cap/" + course + "/" + hand + "/"  + casted_num + ".mp4")
path = "C:/impact_tennis/forehand/all/" + course + "/"

model.classes = [0]

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter(path + casted_num + ".mp4", fourcc, 30, (200,300))

#YOLOv5による選手の検出
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    result = model(frame)
    #result.render()         
    cv2.imshow("test",result.imgs[0])

    MARGIN = 50
        
    for (xmin, ymin, xmax,   ymax,  confidence,  clas) in result.xyxy[0].tolist():
        #選手として認識する位置の制限
        if (xmin > 50 and 1000>ymin>500):
            out.write(frame[int(ymin)-MARGIN:int(ymin)+200+MARGIN,int(xmin):int(xmin)+200])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
out.release()
cv2.destroyAllWindows()

##################################################################################### 

#最後から30フレームを取得する
cap = cv2.VideoCapture("C:/impact_tennis/forehand/all/"+ course + "/" + casted_num + ".mp4")
path = "C:/impact_tennis/forehand/short/"+ course + "/"

total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frame)

cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter(path+casted_num+".mp4",fourcc, fps, (cap_width, cap_height))

count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count>total_frame-31 and count<total_frame :
        cv2.imshow("test",frame)
        writer.write(frame)

    count = count+1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

##################################################################################### 

#BlazePoseで骨格情報を取得する
mp_pose = mp.solutions.pose  
mp_drawing = mp.solutions.drawing_utils 

#BlazePoseを適用する
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #BGR 2 RGB
    image.flags.writeable = False                  
    results = model.process(image)                 #BlazePoseを適用
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #RGB 2 BGR
    return image, results

#使用する骨格パーツの制限
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    pose = np.delete(pose,np.s_[4:44],) #顔のパーツは鼻のみ使用する
    return pose

#推定結果を描画する
def draw_styled_landmarks(image,result):
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)



#numpyファイルの書き込み先
DATA_PATH = "C:/impact_tennis/forehand/MP_Data"

#分類するコース
actions = np.array([str(course)])

#30フレームの動画
sequence_length = 30

os.makedirs(os.path.join(DATA_PATH, str(course), casted_num))  #ファイル作成

cap = cv2.VideoCapture('C:/impact_tennis/forehand/short/'+ course + '/' + casted_num + '.mp4')

#BlazePoseのモデル 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Loop through actions
    for action in actions:
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            #骨格情報を取得する
            image, results = mediapipe_detection(frame, pose)
            
            try:
                landmarks = results.pose_landmarks.landmark 
                print(landmarks)
            except:
                pass
            #骨格情報を描画する
            draw_styled_landmarks(image, results)
            
            #キーポイントの書き込み
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, casted_num, str(frame_num))
            np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
    cap.release()
    cv2.destroyAllWindows()