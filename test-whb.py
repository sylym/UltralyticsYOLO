import asyncio
import datetime
import json
import math
import os
import queue
import subprocess
import threading
import time
from datetime import timedelta
import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error

from cfg.config import TRACKER_CONFIG, MODEL_PATH, MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
from engine.mqtt import MQTTClientHandler
from engine.predictor import YOLOVideoProcessor
from utils.utils import cv2AddChineseText

label_set = [
    [0, 1, 2, 3, 4],
    [5, 6, 7],
    [8, 9],
    [10],
    [11, 12, 13, 14],
    [15],
    [16, 17],
    [18, 19],
    [20],
    [21],
    [22],
    [23],
    [24, 25, 26, 27, 28, 29],
    [30],
    [31, 32, 33, 34, 35, 36],
    [37],
    [38],
]
new2old = {}
for i, l_set in enumerate(label_set):
    for label in l_set:
        new2old[label] = i

label_color = [
    ["FLC1", (255, 128, 0)],
    ["FLA1", (255, 153, 51)],
    ["FLA3", (255, 178, 102)],
    ["FLA4", (230, 230, 0)],
    ["FLA5", (255, 153, 255)],
    ["FLA6", (153, 204, 255)],
    ["FLB1", (255, 102, 255)],
    ["FLB2", (255, 51, 255)],
    ["FLB2", (255, 51, 255)],
    ["FLE1", (51, 153, 255)],
    ["FLE5", (255, 153, 153)],
    ["FLE3", (255, 153, 153)],
    ["FLE4", (255, 51, 51)],
    ["FLE4", (255, 51, 51)],
    ["FLE4", (255, 51, 51)],
    ["FLE4", (255, 51, 51)],
    ["FLF1", (0, 255, 0)],
]

CHNlabel_color = [
    ["锥筒", (255, 128, 0)],
    ["施工标志", (255, 153, 51)],
    ["车道减少标志", (255, 178, 102)],
    ["声光报警器", (230, 230, 0)],
    ["限速标志", (255, 153, 255)],
    ["注意交通引导人员", (153, 204, 255)],
    ["导向标志", (255, 102, 255)],
    ["安全员", (255, 51, 255)],
    ["安全员", (255, 51, 255)],
    ["防撞车", (51, 153, 255)],
    ["汽车", (255, 153, 153)],
    ["汽车", (255, 153, 153)],
    ["人员", (255, 51, 51)],
    ["人员", (255, 51, 51)],
    ["人员", (255, 51, 51)],
    ["人员", (255, 51, 51)],
    ["解除限速标志", (0, 255, 0)],
]

class_mapping_dict = {
        3: 2,
        6: 5, 7: 5,
        9: 8,
        12: 11, 13: 11, 14: 11,
        17: 16,
        19: 18,
        25: 24, 26: 24, 27: 24, 28: 24, 29: 24,
        32: 31, 33: 31, 34: 31, 35: 31, 36: 31
    }



class VideoObjectTracker:
    def __init__(
        self,
        model_path,
        tracker_config,
        output_path,
        mqtt_client,
        source=None,
        zoom=None,
        flight_id=None,
        uav_height=None,
        angle=None,
        rtmp_url=None,
        retry_interval=5,
        max_retry_time=120,

    ):
        self.model = YOLOVideoProcessor(model_path, tracker_config)
        self.source = source  # 可以是视频文件路径或者实时流的URL
        self.output_path = output_path
        self.uav_height = uav_height
        self.angle = angle
        self.video = None
        self.output_video = None
        self.target_id_counter = 0
        self.flc_id_count = [0 for i in range(len(label_color))]  # 保存每个类别的id数量
        self.known_target_id = [
            {} for i in range(len(label_color))
        ]  # 保存每个类别已知的id对应关系
        self.detection_threshold = 0.7
        self.first_distances = {}
        self.distance_threshold = 50
        self.mqtt_client = mqtt_client
        self.flight_id = None
        self.rtmp_url = rtmp_url
        self.width = None
        self.height = None
        self.fps = None
        self.zoom = zoom
        self.ffmpeg_process = None
        self.stop_event = threading.Event()

        self.minio_frame_queue = queue.Queue(maxsize=60)
        self.minio_client = Minio(
            "59.46.115.177:49000",
            access_key="ats",
            secret_key="Abc131419..",
            secure=False,  # set to True if using HTTPS
        )
        self.bucket_name = "ats"

        self.first_detection_time = 0
        self.second_detection_time = 0
        self.third_detection_time = 0
        self.forth_detection_time = 0

        self.total_length = 0
        self.first_length = 0
        self.second_length = 0
        self.thrid_length = 0

        self.retry_interval = retry_interval
        self.max_retry_time = max_retry_time

        self.sent_events = {i: set() for i in range(len(label_color))}

    def open_video(self):
        self.video = cv2.VideoCapture(self.source + "?timeout=5000000")
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.video.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000000)
        if not self.video.isOpened():
            raise ValueError("Cannot open video source")
        self.stop_event.clear()
        self.fps = (
            self.video.get(cv2.CAP_PROP_FPS) or 30
        )  # 如果从实时流读取可能获取不到FPS，可以设定一个默认值
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.VFOV = self.calculate_vfov()
        self.init_rtmp_stream()
        self.init_output_video()

    def calculate_vfov(self):
        focal_length = 9 * self.zoom
        sensor_height = 5.66
        vfov = 2 * math.degrees(math.atan(sensor_height / (2 * focal_length)))
        return vfov

    def init_rtmp_stream(self):
        ffmpeg_command = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            '-re',
            '-threads',
            '5',
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(30),
            "-i",
            "-",
            "-vcodec",
            # "libx264",
            "h264_nvenc",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            "-f",
            "flv",
            self.rtmp_url,
            #"rtmp://59.46.115.177:1935/live/aiOutTest"
        ]
        self.ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    def init_output_video(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.output_video = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

    def save_frame_to_output(self, frame):
        self.output_video.write(frame)

    def stream_frame_to_rtmp(self, frame):
        self.ffmpeg_process.stdin.write(frame.tobytes())

    def publish_event(self, event_info: dict):
        topic = f"/ATS/yunying/task/ai/out/FLXJ/{self.flight_id}"
        try:
            result = self.mqtt_client.publish(topic, json.dumps(event_info), qos=0)
            print(f"Publish result: {result}")
        except Exception as e:
            print(f"Failed to publish event: {e}")

    def stream_frame_to_minio_publish(self, screenshot_path, image_name, event_info):
        if_uploaded = False
        try:
            self.minio_client.fput_object(
                self.bucket_name,
                image_name,
                screenshot_path,
                content_type="image/jpeg"
            )
            if_uploaded = True
            print(f"Successfully uploaded {image_name} to MinIO")
        except S3Error as e:
            print(f"Failed to upload image to MinIO: {e}")
        try:
            url = self.minio_client.presigned_get_object(self.bucket_name, image_name, expires=timedelta(hours=168))
            print(f"Presigned URL: {url}")
        except S3Error as e:
            print("error occurred while generating presigned URL.", e)
        if if_uploaded:
            topic = f"/ATS/yunying/task/ai/out/FLXJ/{self.flight_id}"
            try:
                result = self.mqtt_client.publish(topic, json.dumps(event_info), qos=0)
                print(f"Publish result: {result}")
            except Exception as e:
                print(f"Failed to publish event: {e}")

    def calculate_distances(self, curr_vehicles):
        if len(curr_vehicles) < 2:
            return []
        vehicle_positions = np.array([v["positison"] for v in curr_vehicles])
        distances = np.sqrt(
            (
                (
                    vehicle_positions[:, np.newaxis, :]
                    - vehicle_positions[np.newaxis, :, :]
                )
                ** 2
            ).sum(axis=2)
        )
        consecutive_distances = distances[
            np.arange(len(distances) - 1), np.arange(1, len(distances))
        ]
        return consecutive_distances

    async def process_frames(self):
        retry_time = 0
        while True:
            ret, frame = self.video.read()
            # print(f"第一帧大小：{frame.shape}")
            # if not ret:
            #     break
            print(ret)
            if not ret:
                if retry_time >= self.max_retry_time:
                    print("Exceeded max retry time. Stopping.")
                    break
                try:
                    retry_time += self.retry_interval
                    time.sleep(self.retry_interval)
                    self.video = cv2.VideoCapture(self.source)
                    print(f"Failed to read frame. Retrying in {self.retry_interval} seconds...")
                except Exception as e:
                    print('reconnect failed')
                finally:
                    continue
            retry_time = 0

            self.im0 = frame.copy()
            current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video.get(cv2.CAP_PROP_FPS) or 30
            frame = self.process_frame(
                frame, self.video.get(cv2.CAP_PROP_FRAME_HEIGHT), current_frame, fps
            )
            #frame = cv2.resize(frame, (1920, 1080))

            # 计算距离
            if self.first_detection_time != 0:
                end_time = current_frame / self.fps
                total_time = end_time - self.first_detection_time
                print(f'total_length: {total_time * 5:.2f} m')
                self.total_length = round(total_time * 5)
                cv2.putText(
                        frame,
                        f'total_length:{self.total_length} m',
                        (960, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
            if self.second_detection_time != 0 and self.second_detection_time > self.first_detection_time:
                first_time = self.second_detection_time - self.first_detection_time
                print(f'first_length: {first_time * 5:.2f} m')
                self.first_length = round(first_time * 5)
                cv2.putText(
                        frame,
                        f'first_length:{self.first_length} m',
                        (960, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
            if self.third_detection_time != 0 and self.third_detection_time > self.second_detection_time:
                if self.second_detection_time == 0:
                    second_time = self.third_detection_time - self.first_detection_time
                    print(f'second_length: {second_time * 5:.2f} m')
                    self.second_length = round(second_time * 5)
                    cv2.putText(
                        frame,
                        f'second_length:{self.second_length} m',
                        (960, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                else:
                    second_time = self.third_detection_time - self.second_detection_time
                    print(f'second_length: {second_time * 5:.2f} m')
                    self.second_length = round(second_time * 5)
                    cv2.putText(
                            frame,
                            f'second_length:{self.second_length} m',
                            (960, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
            if self.forth_detection_time != 0 and self.forth_detection_time > self.third_detection_time:
                if self.third_detection_time == 0:
                    third_time = self.forth_detection_time - self.second_detection_time
                    print(f'third_length: {third_time * 5:.2f} m')
                    self.third_length = round(third_time * 5)
                    cv2.putText(
                        frame,
                        f'third_length:{self.third_length} m',
                        (960, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                elif self.second_detection_time == 0:
                    third_time = self.forth_detection_time - self.first_detection_time
                    print(f'third_length: {third_time * 5:.2f} m')
                    self.third_length = round(third_time * 5)
                    cv2.putText(
                            frame,
                            f'third_length:{self.third_length} m',
                            (960, 250),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                else:
                    third_time = self.forth_detection_time - self.third_detection_time
                    print(f'third_length: {third_time * 5:.2f} m')
                    self.third_length = round(third_time * 5)
                    cv2.putText(
                        frame,
                        f'third_length:{self.third_length} m',
                        (960, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
            self.stream_frame_to_rtmp(frame)
            # self.save_frame_to_output(frame)
            print(f"Processing frame {current_frame} of {num_frames}")
            await asyncio.sleep(0)

    def process_frame(self, frame, frame_height, current_frame, fps):
        frame_third = frame_height / 3
        curr_vehicles = []
        else_vehicles = []
        first_distances = self.first_distances
        outputs = self.model.process_frame(frame, slice_height=720, slice_width=720, class_mapping=class_mapping_dict)
        if outputs is None:
            return frame
        if len(outputs) != 0:
            outputs = sorted(outputs, key=lambda x: x[1], reverse=True)
        else:
            return frame
        for output in outputs:
            xmin, ymin, xmax, ymax = map(int, output[:4])
            center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            class_id = new2old[int(output[5])]
            target_id = output[6]
            if target_id is None:
                continue

            if target_id not in self.known_target_id[class_id].keys():
                self.known_target_id[class_id][target_id] = len(
                    self.known_target_id[class_id].keys()
                )
                self.flc_id_count[class_id] += 1
            target_id = self.known_target_id[class_id][target_id]
            class_label, color = label_color[class_id]
            CHNlabel, color2 = CHNlabel_color[class_id]

            if "FLA1" in class_label and self.first_detection_time == 0:
                self.first_detection_time = current_frame / fps
            if "FLA3" in class_label and self.second_detection_time == 0:
                self.second_detection_time = current_frame / fps
            if "FLA6" in class_label and self.third_detection_time == 0:
                self.third_detection_time = current_frame / fps
            if "FLB1" in class_label and self.forth_detection_time == 0:
                self.forth_detection_time = current_frame / fps

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            if class_id == 0:
                curr_vehicles.append(
                    {
                        "target_id": target_id,
                        "class_label": class_label,
                        "class_id": class_id,
                        "position": np.array(
                            [(xmin + xmax) / 2, (ymin + ymax) / 2]
                        ),
                        "center_x": center_x,
                        "center_y": center_y,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                )
            else:
                else_vehicles.append(
                    {
                        "target_id": target_id,
                        "class_label": class_label,
                        "class_id": class_id,
                        "position": np.array(
                            [(xmin + xmax) / 2, (ymin + ymax) / 2]
                        ),
                        "center_x": center_x,
                        "center_y": center_y,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                )
            frame = cv2AddChineseText(frame, f"{CHNlabel} _ {target_id}", (xmin, ymin - 10), (255, 0, 0), 10)
        curr_vehicles.sort(key=lambda v: v["center_y"])
        curr_vehicles = [v for v in curr_vehicles if v["center_y"] > frame_third]
        # distances = self.calculate_distances(curr_vehicles)
        for i in range(len(else_vehicles) - 1):
            vehicle = else_vehicles[i]
            time_ms = round((current_frame / fps) * 1000, 2)
            if vehicle["target_id"] not in self.sent_events[vehicle["class_id"]]:
                screenshot_path = os.path.join(
                    self.screenshot_folder,
                    f'{vehicle["class_label"]}_{vehicle["target_id"]}_{time_ms}.png',
                )
                image_name = (
                    f'{vehicle["class_label"]}_{vehicle["target_id"]}_{time_ms}.png'
                )
                object_name = f"{self.flight_id}/{image_name}"
                cv2.imwrite(screenshot_path, frame)
                if "FLA3" in vehicle["class_label"]:
                    distance = self.first_length * 100
                    event_info = {
                            "event": "meta",
                            "type": vehicle["class_label"],
                            "isWarning": 0,
                            "time": time_ms,
                            "message": None,
                            "info": {
                                "id": vehicle["target_id"],
                                "lon": 125.16,
                                "lat": 41.72,
                                "distance": distance,
                                "same-distance":None,
                            },
                            "url": object_name,
                    }
                elif "FLA6" in vehicle["class_label"]:
                    distance = self.second_length * 100
                    event_info = {
                            "event": "meta",
                            "type": vehicle["class_label"],
                            "isWarning": 0,
                            "time": time_ms,
                            "message": None,
                            "info": {
                                "id": vehicle["target_id"],
                                "lon": 125.16,
                                "lat": 41.72,
                                "distance": distance,
                                "same-distance":None,
                            },
                            "url": object_name,
                    }
                elif "FLB1" in vehicle["class_label"]:
                    distance = self.third_length * 100
                    event_info = {
                            "event": "meta",
                            "type": vehicle["class_label"],
                            "isWarning": 0,
                            "time": time_ms,
                            "message": None,
                            "info": {
                                "id": vehicle["target_id"],
                                "lon": 125.16,
                                "lat": 41.72,
                                "distance": distance,
                                "same-distance":None,
                            },
                            "url": object_name,
                    }
                else:
                    event_info = {
                            "event": "meta",
                            "type": vehicle["class_label"],
                            "isWarning": 0,
                            "time": time_ms,
                            "message": None,
                            "info": {
                                "id": vehicle["target_id"],
                                "lon": 125.16,
                                "lat": 41.72,
                                "distance": None,
                                "same-distance":None,
                            },
                            "url": object_name,
                    }
                threading.Thread(
                    target=self.stream_frame_to_minio_publish,
                    args=(screenshot_path, object_name, event_info),
                ).start()
                self.sent_events[vehicle["class_id"]].add(vehicle["target_id"])
        for i in range(len(curr_vehicles) - 1):
            vehicle = curr_vehicles[i]
            next_vehicle = curr_vehicles[i + 1]
            distance_key = (vehicle["target_id"], next_vehicle["target_id"])
            if distance_key not in first_distances:
                # 计算距离
                distance_pixels = np.sqrt(
                    (vehicle["center_x"] - next_vehicle["center_x"]) ** 2
                    + (vehicle["center_y"] - next_vehicle["center_y"]) ** 2
                )
                distance_meters = distance_pixels * self.get_ratio()
                first_distances[distance_key] = distance_meters
                isWarning = 1 if distance_meters > 4 else 0
                message = "间距过大" if isWarning else "正常"
                # 绘制和显示距离
                cv2.line(
                    frame,
                    (int(vehicle["center_x"]), int(vehicle["center_y"])),
                    (int(next_vehicle["center_x"]), int(next_vehicle["center_y"])),
                    (0, 255, 0),
                    2,
                )
                text = f"{vehicle['class_label']}_{vehicle['target_id']} to {next_vehicle['class_label']}_{next_vehicle['target_id']}: {distance_meters:.2f} m"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[
                    0
                ]
                # 定位文本到每个检测框的右方，稍微偏上一点以避免与检测框重叠
                # 使用xmax和ymax来定位文本，确保文本位于检测框的右侧
                color1 = (0, 0, 255) if distance_meters > 400 else (0, 255, 0)
                text_x = max(int(vehicle["xmax"]), int(next_vehicle["xmax"])) + 50
                text_y = (
                    min(int(vehicle["center_y"]), int(next_vehicle["center_y"]))
                    + 10
                )  # 基于两个检测框的较上方中心点向上偏移10个像素
                # 确保文本位置不会超出图像边界
                # text_x = min(text_x, frame.shape[1] - text_size[0] - 10)
                # text_y = max(text_y, text_size[1] + 10)
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color1,
                    2,
                )
                time_ms = round((current_frame / fps) * 1000, 2)
                # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                screenshot_path = os.path.join(
                    self.screenshot_folder,
                    f'{vehicle["class_label"]}_{vehicle["target_id"]}_{time_ms}.png',
                )
                image_name = (
                    f'{vehicle["class_label"]}_{vehicle["target_id"]}_{time_ms}.png'
                )
                cv2.imwrite(screenshot_path, frame)
                object_name = f"{self.flight_id}/{image_name}"
                if (
                    vehicle["target_id"]
                    not in self.sent_events[vehicle["class_id"]]
                ):
                    distance = round(distance_meters * 100)
                    event_info = {
                        "event": "meta",
                        "type": vehicle["class_label"],
                        "isWarning": isWarning,
                        "time": time_ms,
                        "message": message,
                        "info": {
                            "id": vehicle["target_id"],
                            "lon": 125.16,
                            "lat": 41.72,
                            "distance": distance,
                            "same-distance": distance,
                        },
                        "url": object_name,
                    }
                    threading.Thread(
                        target=self.stream_frame_to_minio_publish,
                        args=(screenshot_path, object_name, event_info),
                    ).start()
                    # threading.Thread(
                    # target=self.publish_event, kwargs={'event_info': event_info}
                    # ).start()
                    self.sent_events[vehicle["class_id"]].add(vehicle["target_id"])
            else:
                distance_meters = first_distances[distance_key]
                # 绘制和显示距离
                cv2.line(
                    frame,
                    (int(vehicle["center_x"]), int(vehicle["center_y"])),
                    (int(next_vehicle["center_x"]), int(next_vehicle["center_y"])),
                    (0, 255, 0),
                    2,
                )
                text = f"{vehicle['class_label']}_{vehicle['target_id']} to {next_vehicle['class_label']}_{next_vehicle['target_id']}: {distance_meters:.2f} m"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[
                    0
                ]
                # 定位文本到每个检测框的右方，稍微偏上一点以避免与检测框重叠
                # 使用xmax和ymax来定位文本，确保文本位于检测框的右侧
                color1 = (0, 0, 255) if distance_meters > 400 else (0, 255, 0)
                text_x = max(int(vehicle["xmax"]), int(next_vehicle["xmax"])) + 50
                text_y = (
                    min(int(vehicle["center_y"]), int(next_vehicle["center_y"]))
                    + 10
                )  # 基于两个检测框的较上方中心点向上偏移10个像素
                # 确保文本位置不会超出图像边界
                #text_x = min(text_x, frame.shape[1] - text_size[0] - 10)
                #text_y = max(text_y, text_size[1] + 10)
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color1,
                    2,
                )
        return frame

    def get_ratio(self):
        equal_length = 35
        return equal_length / int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def close(self):
        if self.video is not None:
            self.video.release()
        if self.output_video is not None:
            self.output_video.release()
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
        cv2.destroyAllWindows()

    def generate_new_paths(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_path = os.path.join("runs", "detect", f"{timestamp}.mp4")
        self.screenshot_folder = os.path.join("runs", "screenshots", timestamp)
        os.makedirs(self.screenshot_folder, exist_ok=True)
        print(
            f"Generated new paths: output_path={self.output_path}, screenshot_folder={self.screenshot_folder}"
        )

    def update_parameters(self, source, uav_height, angle, zoom, flight_id, rtmp_url):
        self.source = source
        self.uav_height = uav_height / 10
        self.angle = -angle / 10
        self.zoom = zoom
        self.flight_id = flight_id
        self.rtmp_url = rtmp_url  # 更新 RTMP 地址
        self.generate_new_paths()  # 生成新的输出路径和截屏路径
        print(
            f"Updated parameters: source={self.source}, uav_height={self.uav_height}, angle={self.angle}, flight_if={self.flight_id}"
        )

    def stop_detection(self):
        self.close()
        self.stop_event.set()
        self.video = None
        self.output_video = None
        self.prev_vehicles = []
        self.target_id_counter = 0
        self.first_distances = {}
        self.sent_events = {i: set() for i in range(len(label_color))}
        self.first_detection_time = 0
        self.second_detection_time = 0
        self.third_detection_time = 0
        self.forth_detection_time = 0
        print("Detection stopped.")


if __name__ == "__main__":
    def on_message(client, topic, payload, qos, properties):
        print(f"Received message: {payload.decode()} on topic {topic}")
        data = json.loads(payload.decode())
        if "param" in data and "resource" in data:
            tracker.update_parameters(
                source=data["resource"],
                uav_height=data["param"]["height"],
                angle=data["param"]["pith"],
                zoom=data["param"]["zoom"],
                flight_id=data["flightId"],
                rtmp_url=data["result"],
            )
            tracker.open_video()
            asyncio.create_task(tracker.process_frames())
        elif "flightId" in data and len(data) == 1:
            tracker.stop_detection()

    client_handler = MQTTClientHandler(MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD, on_message_callback=on_message)

    tracker = VideoObjectTracker(
        model_path=MODEL_PATH,
        tracker_config=TRACKER_CONFIG,
        output_path="",
        mqtt_client=client_handler.client,
        uav_height=80,
        angle=40,
        zoom=2,
        flight_id='4719894fbb634729bd657a69f7963840',
    )

    asyncio.run(client_handler.run())
