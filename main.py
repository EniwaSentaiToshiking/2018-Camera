import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import math
import serial
from timeout_decorator import timeout, TimeoutError
import colorcorrect.algorithm as cca
from PIL import Image
from colorcorrect.util import from_pil, to_pil
import copy
import csv

# 初期化
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

cam_url = 'http://192.168.11.100:8080/?action=stream'  # 0や1でWebCamを指定，当日はURL指定 'http://192.168.11.100:8080/?action=stream'

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# クラス名の読み込み
classesFile = "datasets.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Yolo関連のモデルの読み込み
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# ブロック位置に関する配列の初期化
color_position = []
black_position = []

color_block_position = []
black_block_position = []

send_color_data = []
send_black_data = []

rbyg = []
black = []

pos_array = []

# レイヤーの出力名を取得する
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# 該当するブロックを四角で囲み，ラベルを付ける
def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(im, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    if classes[classId] == 'red':
        rbyg.append([left, top, right, bottom, 1])
    elif classes[classId] == 'blue':
        rbyg.append([left, top, right, bottom, 2])
    elif classes[classId] == 'yellow':
        rbyg.append([left, top, right, bottom, 3])
    elif classes[classId] == 'green':
        rbyg.append([left, top, right, bottom, 4])
    elif classes[classId] == 'black':
        black.append([left, top, right, bottom])

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(im, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(im, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# 前フレームの描画情報を削除し，新しい描画情報に置き換える
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    rbyg.clear()
    black.clear()
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    # Yoloで出力されるボックスの位置を出す
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


# クリックした箇所をサークルと判断する　最初に色付きを16回　後に黒丸を9回
def mouse_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        if len(color_position) <= 15:
            color_position.append([x, y])
        elif len(black_position) <= 8:
            black_position.append([x, y])


# ブロックの位置を取得し，配列に変換する
def get_block_position():
    tmp_color = copy.deepcopy(color_block_position)
    color_block_position.clear()
    send_color_data.clear()
    tmp_black = copy.deepcopy(black_block_position)
    black_block_position.clear()
    send_black_data.clear()

    for i in range(0, len(color_position)):
        for z in range(0, len(rbyg)):
            if rbyg[z][0] <= color_position[i][0] <= rbyg[z][2] and rbyg[z][1] <= color_position[i][1] <= rbyg[z][3]:
                if rbyg[z][4] == 1:
                    color_block_position.append(1)
                elif rbyg[z][4] == 2:
                    color_block_position.append(4)
                elif rbyg[z][4] == 3:
                    color_block_position.append(2)
                elif rbyg[z][4] == 4:
                    color_block_position.append(3)
        if len(color_block_position) <= i:
            color_block_position.append(0)

    for i in range(0, len(black_position)):
        try:
            if black[0][0] <= black_position[i][0] <= black[0][2] and black[0][1] <= black_position[i][1] <= black[0][3]:
                black_block_position.append(1)
            elif black[1][0] <= black_position[i][0] <= black[1][2] and black[1][1] <= black_position[i][1] <= black[1][3]:
                black_block_position.append(1)
            else:
                black_block_position.append(0)
        except:
            black_block_position.append(0)

    # 黄の場所で黄って認識したら赤に変える糞プログラム
    if color_block_position[1] == 2:
        color_block_position[1] = 1
    if color_block_position[7] == 2:
        color_block_position[7] = 1
    if color_block_position[9] == 2:
        color_block_position[9] = 1
    if color_block_position[15] == 2:
        color_block_position[15] = 1

    # 赤の場所で赤って認識したら黄に変える糞プログラム
    if color_block_position[0] == 1:
        color_block_position[0] = 2
    if color_block_position[6] == 1:
        color_block_position[6] = 2
    if color_block_position[8] == 1:
        color_block_position[8] = 2
    if color_block_position[14] == 1:
        color_block_position[14] = 2

    for i in range(1, 5):  # 赤，黄，緑，青の順に配列を整形
        if i in color_block_position:
            if color_block_position.count(i) >= 2:
                send_color_data.append(color_block_position.index(i))
            else:
                send_color_data.append(color_block_position.index(i))
        else:  # もし見つけることが出来なかった場合
            if i in tmp_color:
                send_color_data.append(tmp_color.index(i))  # 一度見つけた色はその色を表示する
                color_block_position[tmp_color.index(i)] = i
            else:
                send_color_data.append(color_block_position.index(0))  # 一度も見つかっていない色は空いている数字にする

    for i in range(0, len(black_block_position)):  # 黒の配列を整形
        if black_block_position[i] == 1:
            send_black_data.append(i)

    if len(send_black_data) < 2:
        if len(tmp_black) == 2:  # 前見つけたデータがあるなら書き換え
            for i in range(0, 2):
                if send_black_data[i] != tmp_black[i]:
                    send_black_data[i] = tmp_black[i]
                    black_block_position[tmp_black[i]] = 1
        else:  # ない場合
            for i in range(0, 2 - len(send_black_data)):
                send_black_data.append(black_block_position.index(0))  # 一度も見つかっていない色は空いている数字にする

    if decode_flag:
        tmp_code_c = pos_array[:4]
        pos_b_code = pos_array[-1]
        error_color = []
        for i in range(0, len(send_color_data)):
            if send_color_data[i] in tmp_code_c:
                tmp_code_c.remove(send_color_data[i])
            elif send_color_data[i] not in tmp_code_c:
                error_color.append(i)

        for i in error_color:
            send_color_data[i] = tmp_code_c.pop(0)

        error_black = []
        tmp_code_b = []

        if pos_b_code == 1:
            tmp_code_b = [0, 4]
        elif pos_b_code == 2:
            tmp_code_b = [1, 5]
        elif pos_b_code == 3:
            tmp_code_b = [3, 7]
        elif pos_b_code == 4:
            tmp_code_b = [4, 8]
        elif pos_b_code == 5:
            tmp_code_b = [1, 3]
        elif pos_b_code == 6:
            tmp_code_b = [2, 4]
        elif pos_b_code == 7:
            tmp_code_b = [4, 6]
        elif pos_b_code == 8:
            tmp_code_b = [5, 7]

        for i in range(0, len(send_black_data)):
            if send_black_data[i] in tmp_code_b:
                tmp_code_b.remove(send_black_data[i])
            elif send_black_data[i] not in tmp_code_b:
                error_black.append(i)

        for i in error_black:
            send_black_data[i] = tmp_code_b.pop(0)


# 取得した画像を合成し，ノイズを取る
def mix_brock():
    img_src1 = cv.imread("./img/1.png", 1)
    img_src2 = cv.imread("./img/2.png", 1)
    img_src3 = cv.imread("./img/3.png", 1)

    try:
        pal = cv.getTrackbarPos('hoge', winName)
    except:
        pal = 150

    img_ave = img_src1 * (1 / 3) + img_src2 * (1 / 3) + img_src3 * (1 / 3) + pal - 150
    cv.imwrite("./img/mix.png", img_ave)
    img = Image.open('./img/mix.png')

    to_pil(cca.grey_world(from_pil(to_pil(cca.stretch(from_pil(img)))))).save('./img/block.png')
    try:
        pal2 = cv.getTrackbarPos('hige', winName)
    except:
        pal2 = 0

    if pal2 == 0:
        to_pil(cca.grey_world(from_pil(to_pil(cca.stretch(from_pil(img)))))).save('./img/block.png')
    elif pal2 == 1:
        to_pil(cca.retinex_with_adjust(cca.retinex(from_pil(img)))).save('./img/block.png')
    elif pal2 == 2:
        to_pil(cca.stretch(from_pil(img))).save('./img/block.png')


@timeout(0.1)
def serial_read(robo):
    return robo.readline().decode('ascii')


def decode_position(pos):
    pos_array.clear()
    for i in list(hex(pos)):
        pos_array.append(i)

    count = 0
    del pos_array[:2]

    for i in pos_array:
        pos_array[count] = int(i, 16)
        count += 1

    if len(pos_array) == 4:
        pos_array.insert(0, 0)


def nothing(val):
    pass


# 描画を行う上での初期設定
winName = 'ET Robo'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.setWindowProperty(winName, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cap = cv.VideoCapture(cam_url)
wname = "MouseEvent"
frame_count = 1

get_block_position_flag = False
decode_flag = False

# BlueToothの初期化
try:
    ser = serial.Serial('/dev/tty.MindstormsEV3-SerialPor', 9600)  # tty.MindstormsEV3-SerialPor or tty.Mindstorms-SerialPortPr
    print('connection done')
    # ser = ''
except :
    ser = ''

while True:
    # ビデオ情報の読み込み
    hasFrame, frame = cap.read()

    # もしビデオカメラの情報がない場合
    if not hasFrame:
        print("エラー：ビデオカメラの情報がありません")
        cv.waitKey(3000)
        break

    cv.imwrite('./img/' + str(frame_count) + '.png', frame)
    frame_count += 1
    if frame_count == 4:
        mix_brock()
        # Yoloを用いたネットワークの構築
        im = cv.imread('./img/block.png')
        blob = cv.dnn.blobFromImage(im, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(im, outs)
        frame_count = 1

        if len(color_position) <= 15:
            text = 'Color Block @%s' % str(16 - len(color_position))
        elif len(black_position) <= 8:
            text = 'Black Block @%s' % str(9 - len(black_position))
        else:
            text = 'Complete'
            get_block_position_flag = True

        if get_block_position_flag:
            get_block_position()

        # ウィンドウの表示
        cv.putText(im, text, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 178, 50), 3)
        for i in color_position:
            cv.ellipse(im, (i[0], i[1]), (5, 5), 0, 0, 360, (0, 0, 255), -1)

        for i in black_position:
            cv.ellipse(im, (i[0], i[1]), (5, 5), 0, 0, 360, (0, 255, 0), -1)

        cv.createTrackbar('hoge', winName, 150, 300, nothing)
        cv.createTrackbar('hige', winName, 0, 2, nothing)
        cv.imshow(winName, im)

    # BlueToothでロボットから送られてくるデータの読み込み
    line = ''
    try:
        line = serial_read(ser)
        print('receive', line)
    except TimeoutError:
        pass
    except:
        pass

    # 四角の描画やマウスイベントの設定
    cv.setMouseCallback(winName, mouse_event)

    # BlueToothで座標データの送信
    if line:  # ロボットからシグナルが来ている場合
        send_data = ''
        print('color', send_color_data)
        print('black', send_black_data)

        # if send_color_data

        # データを2桁に整形
        for val in send_color_data:
            if len(str(val)) == 1:
                send_data += '0' + str(val)
            else:
                send_data += str(val)

        for val in send_black_data:
            send_data += str(val)

        # BlueToothで送信
        ser.write(send_data.encode('ascii'))
        ser.close()

    # ボタン押下時のイベント作成
    key = cv.waitKey(1) & 0xff
    if key == ord('s'):
        with open('color.csv', 'w') as f:
            f1_writer = csv.writer(f, lineterminator='\n')
            f1_writer.writerows(color_position)

        with open('black.csv', 'w') as f:
            f2_writer = csv.writer(f, lineterminator='\n')
            f2_writer.writerows(black_position)
        print('pos save')

    if key == ord('a'):
        print(send_color_data, send_black_data)

    if key == ord('o'):
        f1_o = open('color.csv', 'r')
        f2_o = open('black.csv', 'r')

        for i in f1_o:
            tmp = i.replace('\n', '')
            tmp_2 = tmp.split(",")
            color_position.append([int(tmp_2[0]), int(tmp_2[1])])

        for i in f2_o:
            tmp = i.replace('\n', '')
            tmp_2 = tmp.split(",")
            black_position.append([int(tmp_2[0]), int(tmp_2[1])])

    if key == ord('w'):
        print('Input Initial position code')
        position = int(input())
        decode_position(position)
        decode_flag = True
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    if key == ord('r'):
        ser = serial.Serial('/dev/tty.MindstormsEV3-SerialPor',9600)  # tty.MindstormsEV3-SerialPor or tty.Mindstorms-SerialPortPr
        print('reconnection')

print('end')

# sを押下でクリックした箇所をcsvで出力
# aを押下でBluetoothで送られる内容を表示
# oを押下でクリックした場所を表示
# wを押下で初期位置コードを入力できるように
# rを押下で再度コネクションを張る
# qを押下でシステムを終了させる
