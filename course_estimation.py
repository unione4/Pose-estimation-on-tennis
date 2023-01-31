import os
import numpy as np
import random
import matplotlib.pyplot as plt
import keras

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.math import confusion_matrix
from livelossplot import PlotLossesKeras
from keras.layers import Dense, BatchNormalization, Activation,LayerNormalization


#シード値
fixed_num = 0

#データのパス
DATA_PATH = "C:/impact_tennis/forehand/MP_last_Data/" 

#frames：データのフレーム数，dimensions：データの次元数
frames = 30  
dimensions = 96

#選手から見て3方向への分類
actions = np.array(['opposite','pull','straight'])

label_map = {label:num for num, label in enumerate(actions)} 

print(label_map)

#データ数のカウントを行う
def file_count(file_name):
    initial_count = 0
    dir = DATA_PATH + file_name
    for path in os.listdir(dir):
        initial_count+=1

    return initial_count


#時系列データをラベル付け
sequences, labels = [], []
for action in actions:
    for sequence in range(file_count(action)): 
        window = []
        for frame_num in range(frames):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


X = np.array(sequences)
print(X.shape)
#One-Hotに変換
y = to_categorical(labels).astype(int)
print(y)

#シード値をもとにシャッフル
for l in [X,y]:
    np.random.seed(fixed_num)
    np.random.shuffle(l)

#log_dir = os.path.join(os.getcwd(),'Logs')
#tb_callback = TensorBoard(log_dir=log_dir)


#学習モデル
def build_model():
    model = Sequential()
    model.add(LSTM(512, return_sequences=False, activation='tanh', input_shape=(frames,dimensions))) 
    model.add(BatchNormalization())
    model.add(Dense(256,kernel_initializer='he_normal',activation='relu'))
    model.add(keras.layers.Dropout(rate=0.4))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer=keras.optimizers.RMSprop(decay=0.05), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

#交差検証(10分割)
kf = KFold(n_splits=10, shuffle=False)
all_loss=[]
all_val_loss=[]
all_acc=[]
all_val_acc=[]
all_test_loss = []
all_test_acc = []
recall = []
precision = []
ep=10
for train_index, test_index in kf.split(X,y):

    train_data=X[train_index]
    train_label=y[train_index]
    test_data=X[test_index]
    test_label=y[test_index]

    model=build_model()
    history=model.fit(train_data,train_label,epochs=ep,validation_split=0.1)

    loss=history.history['loss']
    val_loss=history.history['val_loss']
    acc=history.history['categorical_accuracy']
    val_acc=history.history['val_categorical_accuracy']

    pred_y = model.predict(test_data)
    pred_y_classes = np.argmax(pred_y,axis=1)
    answer_label = np.argmax(test_label,axis=1)
    print(pred_y_classes)
    print(answer_label)
    conf = confusion_matrix(answer_label,pred_y_classes)
    print(conf)

    print("precision"+str(precision_score(answer_label,pred_y_classes,average="macro")))
    print("recall" + str(recall_score(answer_label,pred_y_classes,average="macro")))
    test_loss, test_acc = model.evaluate(test_data, test_label, verbose=0)
    print("loss"  + str(test_loss))
    print("acc" + str(test_acc))

    all_loss.append(loss[-1])
    all_val_loss.append(val_loss[-1])
    all_acc.append(acc[-1])
    all_val_acc.append(val_acc[-1])
    recall.append(precision_score(answer_label,pred_y_classes,average="macro"))
    precision.append(recall_score(answer_label,pred_y_classes,average="macro"))
    all_test_loss.append(test_loss)
    all_test_acc.append(test_acc)

#10回分の平均
ave_all_loss = np.mean(all_loss)
ave_all_acc =  np.mean(all_acc)
ave_all_val_loss = np.mean(all_val_loss)
ave_all_val_acc = np.mean(all_val_acc)
ave_test_loss = np.mean(all_test_loss)
ave_test_acc = np.mean(all_test_acc)
ave_recall = np.mean(recall)
ave_precision = np.mean(precision)


#結果の表示
print("train_loss is " +str(ave_all_loss ))
print("train_acc is " +str(ave_all_acc ))
print("val_loss is " +str(ave_all_val_loss ))
print("val_acc is " +str(ave_all_val_acc ))
print("test_loss is " +str(ave_test_loss ))
print("test_acc is " +str(ave_test_acc ))
print("test_recall is " + str(ave_recall))
print("test_precision is " + str(ave_precision))
