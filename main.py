import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# 初期化
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

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

red = []
blue = []
yellow = []
green = []
black = []

# レイヤーの出力名を取得する
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# 該当するブロックを四角で囲み，ラベルを付ける
def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    if classes[classId] == 'red':
        red.append([left, top, right, bottom])
    elif classes[classId] == 'blue':
        blue.append([left, top, right, bottom])
    elif classes[classId] == 'yellow':
        yellow.append([left, top, right, bottom])
    elif classes[classId] == 'green':
        green.append([left, top, right, bottom])
    elif classes[classId] == 'black':
        black.append([left, top, right, bottom])

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# 前フレームの描画情報を削除し，新しい描画情報に置き換える
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
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

    count = 0
    red_flag = False

    #  赤のiの値は0のため，それだけは別に処理
    for i in indices:
        i = i[0]
        count = count + classIds[i]

        if classIds[i] == 0:
            red_flag = True

    # 全色あった場合には書き換えをする．なかった場合は更新を行わない
    if count == 14 and red_flag == True:
        # 配列の初期化
        red.clear()
        blue.clear()
        yellow.clear()
        green.clear()
        black.clear()

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
    color_block_position.clear()
    black_block_position.clear()
    for i in range(0, len(color_position)):
        if red[0][0] <= color_position[i][0] <= red[0][2] and red[0][1] <= color_position[i][1] <= red[0][3]:
            color_block_position.append(1)
        elif blue[0][0] <= color_position[i][0] <= blue[0][2] and blue[0][1] <= color_position[i][1] <= blue[0][3]:
            color_block_position.append(2)
        elif yellow[0][0] <= color_position[i][0] <= yellow[0][2] and yellow[0][1] <= color_position[i][1] <= yellow[0][3]:
            color_block_position.append(3)
        elif green[0][0] <= color_position[i][0] <= green[0][2] and green[0][1] <= color_position[i][1] <= green[0][3]:
            color_block_position.append(4)
        else:
            color_block_position.append(0)

    for i in range(0, len(black_position)):
        if black[0][0] <= black_position[i][0] <= black[0][2] and black[0][1] <= black_position[i][1] <= black[0][3]:
            black_block_position.append(1)
        elif black[1][0] <= black_position[i][0] <= black[1][2] and black[1][1] <= black_position[i][1] <= black[1][3]:
            black_block_position.append(1)
        else:
            black_block_position.append(0)

# 描画を行う上での初期設定
winName = 'ET Robo'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cap = cv.VideoCapture(0)  # 0や1でWebCamを指定，当日はURL指定 'http://192.168.11.100:8080/?action=stream'
wname = "MouseEvent"

while True:
    hasFrame, frame = cap.read()

    # もしビデオカメラの情報がない場合
    if not hasFrame:
        print("エラー：ビデオカメラの情報がありません")
        cv.waitKey(3000)
        break

    # Yoloを用いたネットワークの構築
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    if len(color_position) <= 15:
        text = 'Color Block @%s' % str(16-len(color_position))
    elif len(black_position) <= 8:
        text = 'Black Block @%s' % str(9-len(black_position))
    else:
        text = 'Complete'

    cv.putText(frame, text, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 178, 50), 3)

    # 四角の描画やマウスイベントの設定
    postprocess(frame, outs)
    cv.setMouseCallback(winName, mouse_event)

    # ウィンドウの表示
    cv.imshow(winName, frame)

    # ボタン押下時のイベント作成
    key = cv.waitKey(1) & 0xff
    if key == ord('s'):
        get_block_position()
        print('color', color_block_position)
        print('black', black_block_position)

    elif key == ord('q'):
        cv.destroyAllWindows()
        break


print('end')