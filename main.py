import argparse
from modules.MQTT_processor import CameraImageProcessor
import yaml

if __name__ == '__main__':
    # 创建参数解析器
    args = argparse.ArgumentParser(description='MQTT Broker Information')
    args.add_argument('--config', default='config.yaml', help='Path to the configuration file')
    args.add_argument('--cam', default='1', help='camera number')
    # 解析命令行参数
    args = args.parse_args()

    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    processor = CameraImageProcessor(config, args.cam)
    processor.run()
