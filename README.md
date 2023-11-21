# 第二届日铁杯人脸识别躲避球PK赛

## Dataset

The dataset, containing 536 real face images, can be downloaded by the [link](https://nextcloud01.nssol-sh.com/index.php/s/fCFMkx9w72wmeZ7).

Password: `20231215`

## Environment

Clone the repository first:

`git clone https://github.com/xuejifang/dodgeball.git`

Install the essential packages by:

`pip install -r requirements.txt`

## Config

```yaml
broker_ip: "iot.taginator.cn"   # MQTT broker IP
broker_port: 39683              # MQTT broker port
topic: "nssol/rtb/sensor1"      # MQTT topic, optional "nssol/rtb/sensor1"
```

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