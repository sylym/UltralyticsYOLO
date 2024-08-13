import asyncio
import json
import queue
import subprocess
import threading
import time

import cv2

from cfg.config import MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
from engine.mqtt import MQTTClientHandler


def init_rtmp_stream(width, height, rtmp_url):
    ffmpeg_command = ['ffmpeg',
                      '-y',
                      '-f', 'rawvideo',
                      '-vcodec', 'rawvideo',
                      '-pix_fmt', 'bgr24',
                      '-s', "{}x{}".format(width, height),
                      '-r', '30',
                      '-i', '-',
                      '-c:v', 'libx264',
                      '-pix_fmt', 'yuv420p',
                      '-preset', 'ultrafast',
                      '-f', 'flv',
                      rtmp_url]
    return subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE), ffmpeg_command


class VideoCapture:
    def __init__(self, url, rtmp_url, custom_frame_processor):
        self.URL = url
        self.isstop = False
        self.q = queue.Queue()
        self.capture = cv2.VideoCapture(self.URL)
        while not self.capture.isOpened() and not self.isstop:
            print(f"Unable to open video source {self.URL}. Retrying...")
            time.sleep(1)
            self.capture = cv2.VideoCapture(self.URL)
        self.receive_thread = threading.Thread(target=self.readframe)
        self.process_thread = threading.Thread(target=self.process_frames, args=(custom_frame_processor,))
        self.receive_thread.start()
        self.process_thread.start()
        print('VideoCapture started!')

        (self.ffmpeg_process, self.ffmpeg_command) = init_rtmp_stream(int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                      int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                                      rtmp_url)

    def release(self, custom_release=None):
        self.isstop = True
        if custom_release is not None:
            custom_release()
        print('VideoCapture stopped!')

    def readframe(self):
        while not self.isstop:
            try:
                start = time.time()
                ok, frame = self.capture.read()
                if ok:
                    self.q.put(frame)
                else:
                    end = time.time()
                    if end - start > 20:
                        self.capture.release()
                        self.capture = cv2.VideoCapture(self.URL)
                        while not self.capture.isOpened() and not self.isstop:
                            print(f"Unable to open video source {self.URL}. Retrying...")
                            time.sleep(1)
                            self.capture = cv2.VideoCapture(self.URL)
            except Exception as e:
                print(f"Exception occurred: {e}")
        self.capture.release()
        print("reading stopped!")

    def process_frames(self, frame_callback):
        while not self.isstop or not self.q.empty():
            if self.q.empty():
                continue
            frame = self.q.get_nowait()
            frame = frame_callback(frame)
            if self.ffmpeg_process.poll() is not None:
                self.ffmpeg_process = subprocess.Popen(self.ffmpeg_command, stdin=subprocess.PIPE)
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("Broken pipe error occurred!")
                time.sleep(1)
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.communicate()
        print("processing stopped!")


if __name__ == '__main__':
    cap = None


    def frame_processor(frame):
        return frame


    def on_message(client, topic, payload, qos, properties):
        print(f"Received message: {payload.decode()} on topic {topic}")
        data = json.loads(payload.decode())
        if "param" in data and "resource" in data:
            global cap
            cap = VideoCapture(data["resource"], data["result"], frame_processor)
        elif "flightId" in data and len(data) == 1:
            if cap is not None:
                cap.release()


    client_handler = MQTTClientHandler(MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
                                       on_message_callback=on_message)
    asyncio.run(client_handler.run())
