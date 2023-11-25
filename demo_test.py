import argparse
import base64
import json
import time

import cv2
import paho.mqtt.client as mqtt
import requests
import yaml

from model.detector import Detector


class CameraImageProcessor:
    def __init__(self, config, detector):
        # 获取配置信息
        self.mqtt_broker = config["mqtt"]["broker"]
        self.mqtt_port = config["mqtt"]["port"]
        self.mqtt_topic = config["mqtt"]["topic1"]
        self.http_endpoint = config["http"]["endpoint"]
        self.camera_ip = config["camera"]["ip1"]
        self.rtsp_url = config["camera"]["rtsp_url1"]
        self.team_name = config["team"]["name"]

        # 初始化 rtsp 捕获器
        self.rtsp_cap = cv2.VideoCapture(self.rtsp_url)

        # 初始化 mqtt 客户端
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

        # 初始化检测器
        self.detector = detector

        # 初始化图像配置
        self.image_width = config["image"]["width"]
        self.image_height = config["image"]["height"]

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connect to MQTT Broker")
            # 订阅主题
            self.mqtt_client.subscribe(self.mqtt_topic, qos=0)
            print("MQTT Message Loop is running: ", self.mqtt_client.is_connected())
        else:
            print(f"Failed to connect to MQTT Broker with result code {rc}")

    def on_message(self, client, userdata, msg):
        print("Received MQTT Message ...")

        try:
            start = time.time()
            payload = json.loads(msg.payload.decode("utf-8"))

            if payload["device_id"] == self.camera_ip:
                event_id = payload["event_id"]
                timestamp = payload["timestamp"]

                # 处理当前图像
                ret, image = self.rtsp_cap.read()
                image = cv2.resize(image, (self.image_width, self.image_height))
                result_name, result_action = self.detector.detect(image)

                # 图像编码
                _, img_encoded = cv2.imencode(".jpg", image)
                img_base64 = base64.b64encode(img_encoded).decode("utf-8")

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

                request = requests.post(self.http_endpoint, json=data)
                response = json.loads(request.text)
                response_code = response.get("code")
                response_message = response.get("message")
                print(f"Response Code: {response_code}, \n Response Message: {response_message}")

                end = time.time()
                print("process cost time", end - start)

        except Exception as e:
            print("Error processing MQTT message:", str(e))

    def run(self):
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 5)
        self.mqtt_client.loop_start()

        try:
            while True:
                pass

        except KeyboardInterrupt:
            print("Exiting...")
            self.mqtt_client.disconnect()
            self.mqtt_client.loop_stop()


if __name__ == '__main__':
    # 创建参数解析器
    args = argparse.ArgumentParser(description='MQTT Broker Information')
    args.add_argument('--config', default='config_demo.yaml', help='Path to the configuration file')

    # 解析命令行参数
    args = args.parse_args()

    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    detector = Detector(config)
    print("Init Camera Image Detector Successfully ...")

    processor = CameraImageProcessor(config, detector)
    processor.run()
