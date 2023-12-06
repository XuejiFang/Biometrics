import argparse
import base64
import json
import time

import cv2
import paho.mqtt.client as mqtt
import requests
import yaml
import matplotlib as plt
from model.detector import Detector
import threading
import concurrent.futures
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport"

class RTSCapture(cv2.VideoCapture):
    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"]
    def __init__(self, url, *schemes):
        super().__init__(url)
        # 创建一个读取线程
        self.frame_receiver = threading.Thread(target=self.recv_frame, daemon=True)
        self.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(self.schemes)):
            self._reading = True
        elif isinstance(url, int):
            # 不做任何处理，直接调用父类的方法
            pass
    # 创建RTSCapture对象
    @staticmethod
    def create(url, *schemes):
        return RTSCapture(url, *schemes)
    def isStarted(self):
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok
    # 接收视频流的每一帧
    def recv_frame(self):
        # 当读取标志为True，且视频流打开成功时，循环读取视频帧
        while self._reading and self.isOpened():
            # 调用父类的方法，读取视频帧
            ok, frame = self.read()
            if not ok:
                break
            self._cur_frame = frame
        self._reading = False
    def read2(self):
        # 从类属性中获取最新的一帧
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame
    # 启动读取线程
    def start_read(self):
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read
    def stop_read(self):
        self._reading = False
        if self.frame_receiver.is_alive():
            self.frame_receiver.join()

class CameraImageProcessor:
    def __init__(self, config, thread_name):
        # 获取配置信息
        self.mqtt_broker = config["mqtt"]["broker"]
        self.mqtt_port = config["mqtt"]["port"]
        self.mqtt_topic = config["mqtt"][f"topic{thread_name}"]
        self.http_endpoint = config["http"]["endpoint"]
        self.camera_ip = config["camera"][f"ip{thread_name}"]
        self.rtsp_url = config["camera"][f"rtsp_url{thread_name}"]
        self.team_name = config["team"]["name"]

        # 初始化 rtsp 捕获器
        self.rtsp_cap = RTSCapture.create(self.rtsp_url)
        print(f"Init Camera Image Processor{thread_name} Successfully ...")
        # 启动读取线程
        self.rtsp_cap.start_read()

        # 初始化 mqtt 客户端
        thread_local = threading.local()
        self.mqtt_client = getattr(thread_local, f"mqtt_client_{thread_name}", None)
        if self.mqtt_client is None:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self.on_connect
            self.mqtt_client.on_message = self.on_message
            self.mqtt_client.client_id = f"thread_{thread_name}_client"
            setattr(thread_local, f"mqtt_client_{thread_name}", self.mqtt_client)

        # 初始化图像配置
        self.image_width = config["image"]["width"]
        self.image_height = config["image"]["height"]

        # 初始化检测器

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connect to MQTT Broker")
            # 订阅主题
            self.mqtt_client.subscribe(self.mqtt_topic, qos=0)
            print(f"MQTT Message Loop is running: ", self.mqtt_client.is_connected())
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
                start = time.time()
                time.sleep(0.5)
                ok, image = self.rtsp_cap.read_latest_frame()
                end = time.time()
                print("cost time: ",end - start)
                # 保存图像
                cv2.imwrite(f'./pic/{timestamp}.png', image)
                # image = cv2.resize(image, (self.image_width, self.image_height))  不用resize
                # result_name, result_action = self.detector.detect(image)
                # # 测试代码
                img = cv2.imread('./pic1/7.jpg')
                # testimg = cv2.resize(img, (self.image_width, self.image_height))
                result_name, result_action = self.detector.detect(img)

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
                print(data)
                request = requests.post(self.http_endpoint, json=data)

                response = json.loads(request.text)
                response_code = response.get("code")
                response_message = response.get("message")
                print(f"Response Code: {response_code}, \n Response Message: {response_message}")

                end = time.time()
                print("process cost time", end - start)

        except Exception as e:
            print("Error processing MQTT message:", str(e))

    def on_disconnect(self, client, userdata, rc): # 定义 on_disconnect 函数
        if rc == 0:
            print("Disconnected from MQTT Broker")
        else:
            print(f"Unexpected disconnection from MQTT Broker with result code {rc}")

    def run(self, thread_name):

        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 5)
        print("point1")
        self.mqtt_client.loop_start()
        print("point2")

        thread_local = threading.local()
        self.detector = getattr(thread_local, "detector", None)
        if self.detector is None:
            self.detector = Detector(config)
            setattr(thread_local, "detector", self.detector)

        print(f"Init Camera Image Detector{thread_name} Successfully ...")
        

        try:
            while not exit_event.is_set():
                pass
        finally:
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

    cam1 = CameraImageProcessor(config, 1)
    cam2 = CameraImageProcessor(config, 2)

    exit_event = threading.Event()

    t1 = threading.Thread(target=cam1.run, args=("1",))
    t2 = threading.Thread(target=cam2.run, args=("2",))

    t1.start()
    t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exit_event.set()
        t1.join()
        t2.join()