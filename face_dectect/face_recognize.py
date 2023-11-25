#encoding:utf-8
import argparse
import cv2 as cv
import numpy as np
import time
import pickle
from PIL import Image, ImageDraw, ImageFont

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=10):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "STSONG.TTF", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

def check_act(body_points, left_up_x, left_up_y, right_bottom_x, right_bottom_y):
    poseDict = {0: "举手", 1: "下蹲", 2: "剪刀手", 3: "其他"}
    act = "其他"
    for body_point in body_points:
        if left_up_x<body_point[0]<right_bottom_x and left_up_y<body_point[1]<right_bottom_y:
            act = poseDict[body_point[2]]
    return act

def DrawPicture(image, face_positions, names, body_points, height_radio, width_radio):
    if names:
        for face_position, name in zip(face_positions, names):
            left_up_x = int(face_position[0][0]*width_radio)
            left_up_y = int(face_position[0][1]*height_radio)
            right_bottom_x = int(face_position[1][0]*width_radio)
            right_bottom_y = int(face_position[1][1]*height_radio)
            act = check_act(body_points, left_up_x, left_up_y, right_bottom_x, right_bottom_y)

            text = '人名：%s\n动作：%s\n'%(name[0],act)
            # Draw face bounding box
            cv.rectangle(image, (left_up_x, left_up_y), (right_bottom_x, right_bottom_y), (0, 0, 255), 2)
            # Put Text
            cv.rectangle(image, (left_up_x-10, left_up_y-10), (left_up_x+140, left_up_y-70), (255, 255, 255), 2)
            image = cv2ImgAddText(image, text, left_up_x, left_up_y - 65, (255, 0, 0), 20)
    return image

def Getpos(image, faces):
    output = image.copy()
    position = []

    if faces is not None:
        for idx, face in enumerate(faces):
            coords = face[:-1].astype(np.int32)
            position.append([[coords[0],coords[1]],[coords[0]+coords[2],coords[1]+coords[3]]])
    return position

def DetectFace(image):
    backends = (cv.dnn.DNN_BACKEND_DEFAULT,
                cv.dnn.DNN_BACKEND_HALIDE,
                cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                cv.dnn.DNN_BACKEND_OPENCV)
    targets = (cv.dnn.DNN_TARGET_CPU,
               cv.dnn.DNN_TARGET_OPENCL,
               cv.dnn.DNN_TARGET_OPENCL_FP16,
               cv.dnn.DNN_TARGET_MYRIAD)

    parser = {
        "backend":cv.dnn.DNN_BACKEND_DEFAULT,
        "target":cv.dnn.DNN_TARGET_CPU,
        "model":'./yunet.onnx',
        "score_threshold":0.6,
        "nms_threshold":0.3,
        "top_k":5000
    }
    args = argparse.Namespace(**parser)

    # Instantiate yunet
    yunet = cv.FaceDetectorYN.create(
        model=args.model,
        config='',
        input_size=(320, 320),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        top_k=5000,
        backend_id=args.backend,
        target_id=args.target
    )

    yunet.setInputSize((image.shape[1], image.shape[0]))
    _, faces = yunet.detect(image) # faces: None, or nx15 np.array

    face_position = Getpos(image, faces)
    # for pos in face_position:
    #     cv.imwrite('result.jpg', image[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]])

    return face_position

def ExtractFeature(recognizer, img, face_position):
    features = []
    for pos in face_position:
        face_img = img[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]]
        features.append(recognizer.feature(face_img))
        # rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # features.append(face_recognition.face_encodings(rgb_img)[0])
    return features

def match(recognizer, features, faceDict):
    names = []
    for feature in features:
        maxscore = 0
        for key, value in faceDict.items():
            score = recognizer.match(feature, value, dis_type=0)
            # flag = face_recognition.compare_faces([feature], value)
            # if flag[0]:
            #     names.append([key])
            if abs(score) > maxscore:
                maxscore = abs(score)
                maxscorename = key
        names.append([maxscorename])
    return names

def RecognizeFace(img):
    start_time = time.time()

    #Init
    recognizer = cv.FaceRecognizerSF.create(model='./sface.onnx',
                                            config='',
                                            backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
                                            target_id=cv.dnn.DNN_TARGET_CPU)
    with open('resized_dict_data.pkl', 'rb') as file:
        faceDict = pickle.load(file)
    step1_end_time = time.time()
    height_radio = img.shape[0]/256
    width_radio = img.shape[1]/256
    img = cv.resize(img,(256, 256))

    #Detect
    face_position = DetectFace(img)
    step2_end_time = time.time()

    #Extract Feature
    features = ExtractFeature(recognizer, img, face_position)
    step3_end_time = time.time()

    #Match
    name = match(recognizer, features, faceDict)
    step4_end_time = time.time()

    print(name)

    step1_duration = step1_end_time - start_time
    step2_duration = step2_end_time - step1_end_time
    step3_duration = step3_end_time - step2_end_time
    step4_duration = step4_end_time - step3_end_time
    print("Init:", step1_duration)
    print("Detect:", step2_duration)
    print("Extract Feature:", step3_duration)
    print("Match:", step4_duration)

    return face_position, name, height_radio, width_radio

if __name__ == '__main__':
    # filename = "./facedata/facedata/方学基.png"
    filename = "test3.png"
    image = cv.imdecode(np.fromfile(filename, dtype=np.uint8), cv.IMREAD_COLOR)
    face_position, name, height_radio, width_radio = RecognizeFace(image)  #保留resize缩放比，到draw部分修改框大小，用原图画

    draw_image = DrawPicture(image, face_position, name, [], height_radio, width_radio)
    cv.imwrite('draw.jpg', draw_image)
