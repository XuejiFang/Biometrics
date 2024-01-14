# 第二届日铁杯人脸识别躲避球PK赛
## News
【2023.12.15】We achieved **Runner-Up** in the finals.

<img src="./亚军.jpg" alt="亚军" width="50%">


## Dataset

The dataset, containing 536 real face images, can be downloaded by the [link](https://nextcloud01.nssol-sh.com/index.php/s/fCFMkx9w72wmeZ7).

Password: `20231215`

## Environment

Clone the repository first:

`git clone https://github.com/xuejifang/dodgeball.git`

Install the essential packages by:

`pip install -r requirements.txt`

## Per-trained Models

```shell
mkdir checkpoints
cd checkpoints
```

Download the pre-trained models by the [links](https://westlakeu-my.sharepoint.com/:f:/g/personal/fangxueji_westlake_edu_cn/Eo4icEalRGtIm2qPN_2BSvwBzeFwAw8fA-esv6EDmSE_iA?e=UhMUZA), and put them into the `checkpoints` folder.

## Config

```yaml
broker_ip: "iot.taginator.cn"   # MQTT broker IP
broker_port: 39683              # MQTT broker port
topic: "nssol/rtb/sensor1"      # MQTT topic, optional "nssol/rtb/sensor1"
```

## Face Recognize and Draw Picture

with file `./face_detect/face_recognize.py`

`RecognizeFace()`

    Input: image(numpy array of w*h*3)
    
    Return: face_postion, name, height_ratio, width_ratio
    
        · face_position is a list of detected position of face ([[face1_x, face1_y],[face2_x, face2_y]...])
        
        · name is a list of name of detected face, they are one-to-one correspondence
        
        · height_ratio and width_ratio is the radio between the original pictiure and the resized picture, used for adjusting the detected face bounding box


`DrawPicture()`

    Input: image(original), face_positions, names, body_points(the return of multi-object), height_ratio, width_ratio
    
    Return: a picture with name and action tag over the detected face

      

## To Do

- [x] Subscribe MQTT
  - [ ] Optimize
- [x] Face Detection (YuNet)
- [ ] Face Recognition
- [x] Skeleton Points Detection
- [x] Pose Estimation
  - [ ] Optimize
- [ ] Draw Skeleton and Face Detection Boxes
- [ ] Implement in C++
