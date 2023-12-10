import base64
import json
import time

import cv2
import paho.mqtt.client as mqtt
import requests

from modules.RTSP_cam import RTSPCapture
from modules.detector import Detector


# 结果回传函数
def post_result(data):
    request = requests.post("http://iot.taginator.cn:39699/rtb/result", json=data)
    response = json.loads(request.text)
    response_code = response.get("code")
    response_message = response.get("message")
    print(f"Response Code: {response_code}, \n Response Message: {response_message}")


class CameraImageProcessor:
    def __init__(self, config, cam):
        # 获取配置信息
        self.mqtt_broker = config["mqtt"]["broker"]
        self.mqtt_port = config["mqtt"]["port"]
        self.mqtt_topic = config["mqtt"][f"topic_{cam}"]
        self.http_endpoint = config["http"]["endpoint"]
        self.camera_ip = config["camera"][f"ip_{cam}"]
        self.rtsp_url = config["camera"][f"rtsp_url_{cam}"]
        self.team_name = config["team"]["name"]

        # 初始化 rtsp 捕获器
        self.rtsp_cap = RTSPCapture.create(self.rtsp_url)
        print(f"Init Camera Image Processor Successfully ...")
        # 启动读取线程
        self.rtsp_cap.start_read()

        # 初始化 mqtt 客户端
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

        # 初始化检测器
        self.detector = Detector(config)
        print("Init Camera Image Detector Successfully ...")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            # 订阅主题
            self.mqtt_client.subscribe(self.mqtt_topic, qos=0)
            print(f"MQTT Message Loop is running: ", self.mqtt_client.is_connected())
        else:
            print(f"Failed to connect to MQTT Broker with result code {rc}")

    def on_message(self, client, userdata, msg):
        print("Received MQTT Message ...")

        try:
            payload = json.loads(msg.payload.decode("utf-8"))

            if payload["device_id"] == self.camera_ip:
                event_id = payload["event_id"]
                timestamp = payload["timestamp"]

                # 处理当前图像
                time.sleep(0.5)
                ok, image = self.rtsp_cap.capture()
                # # 保存图像
                # cv2.imwrite(f'./pic/{timestamp}.png', image)
                # # 测试代码
                # img = cv2.imread('./pic1/7.jpg')
                # result_name, result_action, result_img = self.detector.detect(img)
                result_name, result_action, result_img = self.detector.detect(image)

                # 图像编码
                _, img_encoded = cv2.imencode(".jpg", result_img)
                img_base64 = base64.b64encode(img_encoded).decode("utf-8")
                file = open(f'./pic1/{timestamp}.txt', 'w')
                file.write(img_base64)

                # 准备回传数据
                data = {
                    "teamName": self.team_name,
                    "eventId": event_id,
                    "resultImg": f"data:image/jpeg;base64,{img_base64}",
                    "info": []
                }
                for name, action in zip(result_name, result_action):
                    data["info"].append({
                        "resultName": name,
                        "resultAction": action
                    })

                # 回传结果显示
                post_result(data)
        except Exception as e:
            print("Error processing MQTT message:", str(e))

    @staticmethod
    def on_disconnect(self, client, userdata, rc):  # 定义 on_disconnect 函数
        if rc == 0:
            print("Disconnected from MQTT Broker")
        else:
            print(f"Unexpected disconnection from MQTT Broker with result code {rc}")

    def run(self):
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 5)
        self.mqtt_client.loop_forever()
