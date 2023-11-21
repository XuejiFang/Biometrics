import cv2
import numpy as np

mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

nPoints = 18

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
 
colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

def getKeypoints(probMap, threshold=0.1):
 
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
def getValidPairs(output, frameWidth, frameHeight, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))
 
        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
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
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))
 
    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])
 
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

def get_h_w(person_id, position, keypoints_list, personwiseKeypoints):
    h = keypoints_list[int(personwiseKeypoints[person_id][BODY_PARTS[position]])][1]
    w = keypoints_list[int(personwiseKeypoints[person_id][BODY_PARTS[position]])][0]
    return [h, w]

def get_w_h(person_id, position, keypoints_list, personwiseKeypoints):
    return keypoints_list[int(personwiseKeypoints[person_id][BODY_PARTS[position]])]

def if_raising_hands(LEye, REye, LWrist, RWrist):
    if LEye[0] + REye[0] >= LWrist[0] + RWrist[0]:
        return True
    else:
        return False

def if_squaring_hips(LHip, RHip, LKee, RKnee):
    if LHip[0] + RHip[0] >= LKee[0] + RKnee[0]:
        return True
    else:
        return False

def if_scissoring_hands_single(Wrist, top, bottom):
    if top < Wrist[0] and bottom > Wrist[0]:
        return True
    else:
        return False

def if_scissoring_hands(LWrist, RWrist, LShoulder, RShoulder, LHip, RHip, alpha=0.1, beta=0.3):
    top = min(LShoulder[0], RShoulder[0])
    bottom = max(LHip[0], RHip[0])
    alpha = alpha; beta = beta
    top = top + alpha * (bottom - top)
    bottom = bottom - beta * (bottom - top)
    return if_scissoring_hands_single(LWrist, top, bottom) or if_scissoring_hands_single(RWrist, top, bottom)

def get_pose(net, image1, inHeight):
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]
    inWidth = int((inHeight/frameHeight)*frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    detected_keypoints = []
    keypoint_id = 0
    keypoints_list = np.zeros((0,3))
    threshold = 0.1
    
    for part in range(nPoints):
        probMap = output[0,part,:,:]
        # 这里可以优化
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1
    
        detected_keypoints.append(keypoints_with_id)

    frameClone = image1.copy()
    for i in range(nPoints):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
    cv2.imwrite("./results/Keypoints.jpg",frameClone)
    
    valid_pairs, invalid_pairs = getValidPairs(output, frameWidth, frameHeight, detected_keypoints)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
    cv2.imwrite("./results/DetectedPose.jpg" , frameClone)

    pose_list = []
    for person in personwiseKeypoints:
        i = int(person[0])
        if i < 0:
            continue
        if if_raising_hands(get_h_w(i, 'LEye', keypoints_list, personwiseKeypoints), get_h_w(i, 'REye', keypoints_list, personwiseKeypoints), get_h_w(i, 'LWrist', keypoints_list, personwiseKeypoints), get_h_w(i, 'RWrist', keypoints_list, personwiseKeypoints)):
            pose = 0
        elif if_squaring_hips(get_h_w(i, 'LHip', keypoints_list, personwiseKeypoints), get_h_w(i, 'RHip', keypoints_list, personwiseKeypoints), get_h_w(i, 'LKnee', keypoints_list, personwiseKeypoints), get_h_w(i, 'RKnee', keypoints_list, personwiseKeypoints)):
            pose = 1
        elif if_scissoring_hands(get_h_w(i, 'LWrist', keypoints_list, personwiseKeypoints), get_h_w(i, 'RWrist', keypoints_list, personwiseKeypoints), get_h_w(i, 'LShoulder', keypoints_list, personwiseKeypoints), get_h_w(i, 'RShoulder', keypoints_list, personwiseKeypoints), get_h_w(i, 'LHip', keypoints_list, personwiseKeypoints), get_h_w(i, 'RHip', keypoints_list, personwiseKeypoints)):
            pose = 2
        else:
            pose = 3
        nose_w_h = get_w_h(i, 'Nose', keypoints_list, personwiseKeypoints)
        pose_list.append([nose_w_h[0], nose_w_h[1], pose])
        """
        print('Person', i)
        print(if_raising_hands(get_h_w(i, 'LEye', keypoints_list, personwiseKeypoints), get_h_w(i, 'REye', keypoints_list, personwiseKeypoints), get_h_w(i, 'LWrist', keypoints_list, personwiseKeypoints), get_h_w(i, 'RWrist', keypoints_list, personwiseKeypoints)))   
        print(if_squaring_hips(get_h_w(i, 'LHip', keypoints_list, personwiseKeypoints), get_h_w(i, 'RHip', keypoints_list, personwiseKeypoints), get_h_w(i, 'LKnee', keypoints_list, personwiseKeypoints), get_h_w(i, 'RKnee', keypoints_list, personwiseKeypoints)))
        print(if_scissoring_hands(get_h_w(i, 'LWrist', keypoints_list, personwiseKeypoints), get_h_w(i, 'RWrist', keypoints_list, personwiseKeypoints), get_h_w(i, 'LShoulder', keypoints_list, personwiseKeypoints), get_h_w(i, 'RShoulder', keypoints_list, personwiseKeypoints), get_h_w(i, 'LHip', keypoints_list, personwiseKeypoints), get_h_w(i, 'RHip', keypoints_list, personwiseKeypoints)))    
        print(nose_w_h)
        print('---------------------')     
        """

    return frameClone, pose_list