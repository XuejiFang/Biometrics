import cv2
import numpy as np
import pickle
import torch.nn as nn
import torch
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 512)
        self.linear7 = nn.Linear(512, 536)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        feature = torch.tensor(feature)
        feature = feature.view((1, 128))
        feature = self.linear1(feature)
        feature = self.relu(feature)
        feature = self.linear4(feature)
        feature = self.relu(feature)
        feature = self.linear7(feature)
        feature = self.sigmoid(feature)

        return feature

class Detector:
    def __init__(self, config):
        self.protoFile = config['pose']["protoFile"]
        self.weightsFile = config['pose']["weightsFile"]

        self.yunetFile = config["yunet"]["yunetFile"]
        self.sfaceFile = config['yunet']["sfaceFile"]
        self.modelFile = config['yunet']["modelFile"]
        self.encoderFile = config['yunet']["encoderFile"]
        self.score_threshold = config['yunet']['score_threshold']
        self.nms_threshold = config['yunet']['nms_threshold']

        self.image_width = config["image"]["width"]
        self.image_height = config["image"]["height"]

        self.textSize = config["textSize"]
        self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        self.target = cv2.dnn.DNN_TARGET_CPU

        self.yunet = cv2.FaceDetectorYN.create(
            model=self.yunetFile,
            config='',
            input_size=(self.image_height, self.image_width),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k = 5000,
            backend_id=self.backend,
            target_id=self.target
        )

        self.recognizer = cv2.FaceRecognizerSF.create(model=self.sfaceFile,
                                        config='',
                                        backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                                        target_id=cv2.dnn.DNN_TARGET_CPU)
        self.model = Network()
        self.model.load_state_dict(torch.load(self.modelFile, map_location=torch.device('cpu')))
        with open(self.encoderFile, 'rb') as file:
            self.name_encoder = pickle.load(file)

        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
        self.mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
                [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
                [47,48], [49,50], [53,54], [51,52], [55,56],
                [37,38], [45,46]]

        self.POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                    [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                    [1,0], [0,14], [14,16], [0,15], [15,17],
                    [2,17], [5,16] ]

        self.nPoints = 18

        self.BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
        
        self.colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
                [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
                [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]
    
    def getKeypoints(self, probMap, threshold=0.1):
    
        mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
    
        mapMask = np.uint8(mapSmooth>threshold)
        keypoints = []
    
        #find the blobs
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        #for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    
        return keypoints
    
    
    # Find valid connections between the different joints of a all persons present
    def getValidPairs(self, output, frameWidth, frameHeight, detected_keypoints):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        # loop for every POSE_PAIR
        for k in range(len(self.mapIdx)):
            # A->B constitute a limb
            pafA = output[0, self.mapIdx[k][0], :, :]
            pafB = output[0, self.mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (frameWidth, frameHeight))
            pafB = cv2.resize(pafB, (frameWidth, frameHeight))
    
            # Find the keypoints for the first and second limb
            candA = detected_keypoints[self.POSE_PAIRS[k][0]]
            candB = detected_keypoints[self.POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)
    
            # If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid
    
            if( nA != 0 and nB != 0):
                valid_pair = np.zeros((0,3))
                for i in range(nA):
                    max_j=-1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                            pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores)/len(paf_scores)
    
                        # Check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                        if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)
    
                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else: # If no keypoints are detected
                print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs
    

    # This function creates a list of keypoints belonging to each person
    # For each detected valid pair, it assigns the joint(s) to a person
    def getPersonwiseKeypoints(self, valid_pairs, invalid_pairs, keypoints_list):
        # the last number in each row is the overall score
        personwiseKeypoints = -1 * np.ones((0, 19))
    
        for k in range(len(self.mapIdx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:,0]
                partBs = valid_pairs[k][:,1]
                indexA, indexB = np.array(self.POSE_PAIRS[k])
    
                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break
    
                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]
    
                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints

    def get_h_w(self, person_id, position, keypoints_list, personwiseKeypoints):
        h = keypoints_list[int(personwiseKeypoints[person_id][self.BODY_PARTS[position]])][1]
        w = keypoints_list[int(personwiseKeypoints[person_id][self.BODY_PARTS[position]])][0]
        return [h, w]

    def get_w_h(self, person_id, position, keypoints_list, personwiseKeypoints):
        return keypoints_list[int(personwiseKeypoints[person_id][self.BODY_PARTS[position]])]

    def if_raising_hands(self, LEye, REye, LWrist, RWrist):
        if LEye[0] + REye[0] >= LWrist[0] + RWrist[0]:
            return True
        else:
            return False

    def if_squaring_hips(self, LHip, RHip, LKee, RKnee):
        if LHip[0] + RHip[0] >= LKee[0] + RKnee[0]:
            return True
        else:
            return False

    def if_scissoring_hands_single(self, Wrist, top, bottom):
        if top < Wrist[0] and bottom > Wrist[0]:
            return True
        else:
            return False

    def if_scissoring_hands(self, LWrist, RWrist, LShoulder, RShoulder, LHip, RHip, alpha=0.1, beta=0.3):
        top = min(LShoulder[0], RShoulder[0])
        bottom = max(LHip[0], RHip[0])
        alpha = alpha; beta = beta
        top = top + alpha * (bottom - top)
        bottom = bottom - beta * (bottom - top)
        return self.if_scissoring_hands_single(LWrist, top, bottom) or self.if_scissoring_hands_single(RWrist, top, bottom)

    def get_pose(self, image1, inHeight, if_save=False):
        frameWidth = image1.shape[1]
        frameHeight = image1.shape[0]
        inWidth = int((inHeight/frameHeight)*frameWidth)

        inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inpBlob)
        output = self.net.forward()

        detected_keypoints = []
        keypoint_id = 0
        keypoints_list = np.zeros((0,3))
        threshold = 0.1
        
        for part in range(self.nPoints):
            probMap = output[0,part,:,:]
            # 这里可以优化
            probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
            keypoints = self.getKeypoints(probMap, threshold)
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1
        
            detected_keypoints.append(keypoints_with_id)

        frameClone = image1.copy()
        for i in range(self.nPoints):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, self.colors[i], -1, cv2.LINE_AA)
        if if_save: cv2.imwrite("./results/Keypoints.jpg",frameClone)
        
        valid_pairs, invalid_pairs = self.getValidPairs(output, frameWidth, frameHeight, detected_keypoints)
        personwiseKeypoints = self.getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(self.POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)
        if if_save: cv2.imwrite("./results/DetectedPose.jpg" , frameClone)

        pose_list = []
        for person in personwiseKeypoints:
            i = int(person[0])
            pose = []
            if i < 0:
                continue
            if self.if_raising_hands(self.get_h_w(i, 'LEye', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'REye', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'LWrist', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'RWrist', keypoints_list, personwiseKeypoints)):
                pose.append(0)
            elif self.if_squaring_hips(self.get_h_w(i, 'LHip', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'RHip', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'LKnee', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'RKnee', keypoints_list, personwiseKeypoints)):
                pose.append(1)
            elif self.if_scissoring_hands(self.get_h_w(i, 'LWrist', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'RWrist', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'LShoulder', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'RShoulder', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'LHip', keypoints_list, personwiseKeypoints), self.get_h_w(i, 'RHip', keypoints_list, personwiseKeypoints)):
                pose.append(2)

            nose_w_h = self.get_w_h(i, 'Nose', keypoints_list, personwiseKeypoints)
            pose_list.append([nose_w_h[0], nose_w_h[1], pose])


        return frameClone, pose_list, detected_keypoints, keypoints_list, personwiseKeypoints

    # face detection
    def Getpos(self, image, faces):
        output = image.copy()
        position = []

        if faces is not None:
            for idx, face in enumerate(faces):
                coords = face[:-1].astype(np.int32)
                position.append([[coords[0],coords[1]],[coords[0]+coords[2],coords[1]+coords[3]]])
        return position

    def DetectFace(self, image):
        # backends = (cv2.dnn.DNN_BACKEND_DEFAULT,
        #             cv2.dnn.DNN_BACKEND_HALIDE,
        #             cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
        #             cv2.dnn.DNN_BACKEND_OPENCV)
        # targets = (cv2.dnn.DNN_TARGET_CPU,
        #         cv2.dnn.DNN_TARGET_OPENCL,
        #         cv2.dnn.DNN_TARGET_OPENCL_FP16,
        #         cv2.dnn.DNN_TARGET_MYRIAD)

        self.yunet.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.yunet.detect(image) # faces: None, or nx15 np.array

        face_position = self.Getpos(image, faces)

        return face_position

    def ExtractFeature(self, recognizer, img, face_position):
        features = []
        for pos in face_position:
            face_img = img[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]]
            features.append(recognizer.feature(face_img))
        return features

    def recognize(self, model, features, name_encoder):
        names = []
        for feature in features:
            pred = model(feature)
            index = np.argmax(pred.detach().numpy())
            names.append(name_encoder.inverse_transform([index]))
        return names

    def padding(self, img):   #padding到4：3
        width = img.shape[1]/4
        height = img.shape[0]/3
        if width == height: return img, -1, 0
        if width > height: flag = 1 #只需要在高上padding
        else: flag = 0  #只需要在宽上padding

        #计算补充量
        if flag == 0:
            delta = height * 4 - img.shape[1]
            pad_img = cv2.copyMakeBorder(img, 0, 0, int(delta//2), int(delta-delta//2), cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255))
        else:
            delta = width * 3 - img.shape[0]
            pad_img = cv2.copyMakeBorder(img, 0, 0, int(delta // 2), int(delta - delta // 2), cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255))
        return pad_img, flag, int(delta // 2)

    def RecognizeFace(self, img):
        #Init
        padding_img, flag, padding_len = self.padding(img)
        #Detect
        height_ratio = padding_img.shape[0] / 480
        width_ratio = padding_img.shape[1] / 640
        resized_img = cv2.resize(padding_img, (640, 480))
        face_position = self.DetectFace(resized_img)

        #Extract Feature
        features = self.ExtractFeature(self.recognizer, resized_img, face_position)
        #Match
        name = self.recognize(self.model, features, self.name_encoder)

        print(name)
        for pos in face_position:
            pos[0][0] = int(pos[0][0] * width_ratio)
            pos[0][1] = int(pos[0][1] * height_ratio)
            pos[1][0] = int(pos[1][0] * width_ratio)
            pos[1][1] = int(pos[1][1] * height_ratio)
            if flag != -1:
                pos[0][0] = int(pos[0][0] - padding_len)
                pos[1][0] = int(pos[1][0] - padding_len)
            pos[0][0] = int(pos[0][0] / img.shape[1] * 640)
            pos[1][0] = int(pos[1][0] / img.shape[1] * 640)
            pos[0][1] = int(pos[0][1] / img.shape[0] * 480)
            pos[1][1] = int(pos[1][1] / img.shape[0] * 480)
        return face_position, name
    
    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=10):
        if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "./fonts/STSong.TTF", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def check_act(self, body_points, left_up_x, left_up_y, right_bottom_x, right_bottom_y):
        # poseDict = {0: "举手", 1: "下蹲", 2: "剪刀手", 3: "其他"}
        poseDict = ["举手", "下蹲", "剪刀手", "其他"]
        # act = "其他"
        act_list = []
        for body_point in body_points:
            if left_up_x<body_point[0]<right_bottom_x and left_up_y<body_point[1]<right_bottom_y:
                for index in body_point[2]:
                    act = poseDict[index]
                    act_list.append(act)
        return act_list if act_list else ["其他"]


    def DrawPicture(self, image, face_positions, names, body_points, detected_keypoints, keypoints_list, personwiseKeypoints):
        image = image.copy()
        name_list = []
        act_list = []
        for i in range(self.nPoints):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(image, detected_keypoints[i][j][0:2], 5, self.colors[i], -1, cv2.LINE_AA)
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(self.POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(image, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)
        if names:
            for face_position, name in zip(face_positions, names):
                left_up_x = face_position[0][0]
                left_up_y = face_position[0][1]
                right_bottom_x = face_position[1][0]
                right_bottom_y = face_position[1][1]
                act = self.check_act(body_points, left_up_x, left_up_y, right_bottom_x, right_bottom_y)
                

                text = '人名：%s\n动作：%s\n'%(name[0],act)
                name_list.append(name[0])
                act_list.append(act)

                # Draw face bounding box
                cv2.rectangle(image, (left_up_x, left_up_y), (right_bottom_x, right_bottom_y), (0, 0, 255), 2)
                # Put Text
                cv2.rectangle(image, (left_up_x-10, left_up_y-10), (left_up_x+140, left_up_y-70), (255, 255, 255), 2)
                image = self.cv2ImgAddText(image, text, left_up_x, left_up_y - 65, (255, 0, 0), self.textSize)
        return image, name_list, act_list
    
    def detect(self, img):
        resized_img = cv2.resize(img, (640, 480))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 并行执行 get_pose 和 RecognizeFace
            pose_future = executor.submit(self.get_pose, resized_img, 320, if_save=True)
            face_future = executor.submit(self.RecognizeFace, img)

            # 获取 get_pose 的结果
            image_pose, pose_list, detected_keypoints, keypoints_list, personwiseKeypoints = pose_future.result()

            # 获取 RecognizeFace 的结果
            face_position, name = face_future.result()

            # 继续执行剩余部分的代码
            draw_image, name_list, act_list = self.DrawPicture(cv2.resize(resized_img, (640,480)), face_position, name, pose_list, detected_keypoints, keypoints_list, personwiseKeypoints)

            cv2.imwrite('./results/draw.jpg', draw_image)

            return name_list, act_list
