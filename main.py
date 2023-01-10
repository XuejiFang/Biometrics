import os
import argparse

import numpy as np
import cv2 as cv

# for handPose
from HandTrackingModule import HandDetector
import random


parser = argparse.ArgumentParser()
parser.add_argument('--database_dir', '-db', type=str, default='./database/for_train/')
parser.add_argument('--face_detection_model', '-fd', type=str,
                    default='./models/face_detection_yunet_2021dec-quantized.onnx', required=False)
parser.add_argument('--face_recognition_model', '-fr', type=str,
                    default='./models/face_recognition_sface_2021dec-quantized.onnx', required=False)
args = parser.parse_args()


def detect_face(detector, image):
    ''' Run face detection on input image.

    Paramters:
        detector - an instance of cv.FaceDetectorYN
        image    - a single image read using cv.imread

    Returns:
        faces    - a np.array of shape [n, 15]. If n = 0, return an empty list.
    '''
    faces = []
    ### TODO: your code starts here

    # 检测人脸
    results = detector.detect(image)
    if isinstance(results, tuple):
        faces = results[1] if results[1] is not None else [];
    elif isinstance(results, list):
        faces = [results[fi][1] for fi in len(results) if results[fi][1] is not None]
    else:
        faces = []
    ### your code ends here
    return faces


def extract_feature(recognizer, image, faces):
    ''' Run face alignment on the input image & face bounding boxes; Extract features from the aligned faces.

    Parameters:
        recognizer - an instance of cv.FaceRecognizerSF
        image      - a single image read using cv.imread
        faces      - the return of detect_face

    Returns:
        features   - a length-n list of extracted features. If n = 0, return an empty list.
    '''
    features = []
    ### TODO: your code starts here

    for i in range(len(faces)):
        aligned_face = recognizer.alignCrop(image, faces[i][:-1])
        features.append(recognizer.feature(aligned_face))

    ### your code ends here
    return features


def match(recognizer, feature1, feature2, dis_type=1):
    ''' Calculate the distatnce/similarity of the given feature1 and feature2.

    Parameters:
        recognizer - an instance of cv.FaceRecognizerSF. Call recognizer.match to calculate distance/similarity
        feature1   - extracted feature from identity 1
        feature2   - extracted feature from identity 2
        dis_type   - 0: cosine similarity; 1: l2 distance; others invalid

    Returns:
        isMatched  - True if feature1 and feature2 are the same identity; False if different
    '''
    l2_threshold = 1.128
    cosine_threshold = 0.363
    isMatched = False
    ### TODO: your code starts here

    if dis_type == 0:
        score = recognizer.match(feature1, feature2, dis_type=0)
        if score < cosine_threshold:
            isMatched = 1
    else:
        score = recognizer.match(feature1, feature2, dis_type=1)
        if score < l2_threshold:
            isMatched = 1

    ### your code ends here
    return isMatched


def get_identity(filename):
    # original
    # identity = filename[:-4]
    filename = str.split(filename, '.')[0]
    identity = str.split(filename, '_')[0]

    return identity


def load_database(database_path, detector, recognizer):
    ''' Load database from the given database_path into a dictionary. It tries to load extracted features first, and call detect_face & extract_feature to get features from images (*.jpg, *.png).

    Parameters:
        database_path - path to the database directory
        detector      - an instance of cv.FaceDetectorYN
        recognizer    - an instance of cv.FaceRecognizerSF

    Returns:
        db_features   - a dictionary with filenames as key and features as values. Keys are used as identity.
    '''
    db_features = dict()

    print('Loading database ...')
    # load pre-extracted features first
    for filename in os.listdir(database_path):
        if filename.endswith('.npy'):
            # identity = filename[:-4]
            identity = str.split(filename, '.')[0]
            if identity not in db_features:
                db_features[identity] = np.load(os.path.join(database_path, filename))
    npy_cnt = len(db_features)
    # load images and extract features
    for filename in os.listdir(database_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # identity = filename[:-4]
            identity = get_identity(filename)
            print(identity)
            if identity not in db_features:
                image = cv.imread(os.path.join(database_path, filename))
                faces = detect_face(detector, image)
                features = extract_feature(recognizer, image, faces)
                if len(features) > 0:
                    db_features[identity] = features[0]
                    np.save(os.path.join(database_path, '{}.npy'.format(identity)), features[0])
    cnt = len(db_features)
    print(
        'Database: {} loaded in total, {} loaded from .npy, {} loaded from images.'.format(cnt, npy_cnt, cnt - npy_cnt))
    return db_features


def visualize(image, faces, identities, fps, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    output = image.copy()

    # put fps in top-left corner
    cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    for face, identity in zip(faces, identities):
        # draw bounding box
        bbox = face[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), box_color, 2)
        # put identity
        cv.putText(output, '{}'.format(identity), (bbox[0], bbox[1] - 15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    return output

# 改变图片的亮度与对比度

def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    # image = []
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img

def getFace(detector, target_size, output_dir=None):
    # 使用dlib自带的frontal_face_detector作为我们的特征提取器
    # detector = dlib.get_frontal_face_detector()
    # 打开摄像头 参数为输入流，可以为摄像头或视频文件
    camera = cv.VideoCapture(0)

    name = input("请输入录入人的姓名:")
    index = 1
    ok = True

    while ok:

        # 从摄像头读取照片
        # 读取摄像头中的图像，ok为是否读取成功的判断参数
        ok, img = camera.read()

        # 转换成灰度图像
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        img = cv.resize(img, (target_size[0], target_size[1]))
        dets = detector.detect(img)

        # 展示摄像头读取到的图片
        cv.imshow(name, img)
        key = cv.waitKey(1)
        # ESC退出
        if key == 27:
            cv.destroyAllWindows()
            return 0
        # s保存
        elif key == 115:
            # 保存图片
            cv.imwrite(os.path.join(args.database_dir, name + '_' + str(index-1) + '.jpg'), img)
            print("save success ")
            index += 1


if __name__ == '__main__':
    target_size = [640, 480]
    # Initialize FaceDetectorYN
    detector = cv.FaceDetectorYN.create(model=args.face_detection_model,
                                        config='',
                                        input_size=target_size,
                                        score_threshold=0.999,
                                        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
                                        target_id=cv.dnn.DNN_TARGET_CPU
                                        )
    # Initialize FaceRecognizerSF
    recognizer = cv.FaceRecognizerSF.create(model=args.face_recognition_model,
                                            config='',
                                            backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
                                            target_id=cv.dnn.DNN_TARGET_CPU
                                            )
    # detect hand
    detect_hand = HandDetector()

    # Load database
    database = load_database(args.database_dir, detector, recognizer)

    # Initialize video stream
    device_id = 0
    cap = cv.VideoCapture(device_id)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Real-time face recognition
    tm = cv.TickMeter()
    enterNameFlag = 0
    while True:
        key = cv.waitKey(1)
        if key == 27:
            cv.destroyAllWindows()
            break
        hasFrame, frame = cap.read()
        try:
            frame = cv.resize(frame, (target_size[0], target_size[1]))
            raw_frame = np.copy(frame)
        except:
            continue
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()

        # detect hand pose
        hands = detect_hand.findHands(frame)
        lmList, bbox = detect_hand.findPosition(hands)

        if lmList:
            x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
            x1, x2, x3, x4, x5 = detect_hand.fingersUp()

            if (x2 == 1 and x3 == 1) and (x4 == 0 and x5 == 0 and x1 == 0):
                cv.putText(hands, "2_TWO", (x_1, y_1), cv.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif (x2 == 1 and x3 == 1 and x4 == 1) and (x1 == 0 and x5 == 0):
                cv.putText(hands, "3_THREE", (x_1, y_1), cv.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif (x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1) and (x1 == 0):
                cv.putText(hands, "4_FOUR", (x_1, y_1), cv.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif x1 == 1 and x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1:
                cv.putText(hands, "5_FIVE", (x_1, y_1), cv.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif x2 == 1 and (x1 == 0, x3 == 0, x4 == 0, x5 == 0):
                cv.putText(hands, "1_ONE", (x_1, y_1), cv.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif x1 and (x2 == 0, x3 == 0, x4 == 0, x5 == 0):
                cv.putText(hands, "Press [y/Y] If You", (x_1, y_1 ), cv.FONT_HERSHEY_PLAIN, 2,
                            (0, 255, 255), 2)
                cv.putText(hands, "Want To Register/Update", (x_1, y_1 + 30), cv.FONT_HERSHEY_PLAIN, 2,
                                            (0, 255, 255), 2)
                if ( key == 121 or key == 89 ):
                    cv.putText(frame, "Please Enter Face Name In Terminal ", (30, 60), cv.FONT_HERSHEY_PLAIN, 2,
                                    (255, 255, 0), 2)
                    enterNameFlag = 2

        
        # detect faces
        faces = detect_face(detector, frame)

        while(enterNameFlag == 1):
            name = input("请输入录入人的姓名:")
            if(len(name) > 0):
                print(name)
                enterNameFlag = 0;
                cv.imwrite(os.path.join(args.database_dir, name + '.png'), raw_frame)
                database = load_database(args.database_dir, detector, recognizer)
                break
        if(enterNameFlag):
            enterNameFlag = 1
            
        # extract features
        features = extract_feature(recognizer, frame, faces)
        # match detected faces with database
        identities = []
        for feature in features:
            isMatched = False
            for identity, db_feature in database.items():
                isMatched = match(recognizer, feature, db_feature)
                if isMatched:
                    identities.append(identity)
                    break
            if not isMatched:
                identities.append('Unknown')
        tm.stop()

        # Draw results on the input image
        frame = visualize(frame, faces, identities, tm.getFPS())

        # Visualize results in a new Window
        cv.imshow('Face recognition system', frame)

        tm.reset()
