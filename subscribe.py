import argparse
import yaml

import numpy as np
import cv2
import paho.mqtt.client as mqtt

# 创建参数解析器
args = argparse.ArgumentParser(description='MQTT Broker Information')
args.add_argument('--config', default='config.yaml', help='Path to the configuration file')

# 解析命令行参数
args = args.parse_args()

# 读取配置文件
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
# 获取参数值
broker_ip = config['broker_ip']
broker_port = config['broker_port']
topic = config['topic']

# MQTT连接回调函数
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    # 在连接成功后订阅主题
    client.subscribe(topic)

# MQTT消息接收回调函数
def on_message(client, userdata, msg):
    try:
        # 将接收到的消息转换为图像数组
        nparr = np.frombuffer(msg.payload, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # 显示图像
        cv2.imshow("Received Image", image)
        cv2.waitKey(1)
    except Exception as e:
        print("Error displaying image: " + str(e))

# 创建MQTT客户端并设置回调函数
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# 连接到MQTT代理
client.connect(broker_ip, broker_port, 60)

# 循环监听MQTT消息
client.loop_forever()