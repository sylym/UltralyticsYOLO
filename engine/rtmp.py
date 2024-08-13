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
        self.isstop = False
        self.q = queue.Queue()
        self.init_event = threading.Event()
        self.size = None
        self.receive_thread = threading.Thread(target=self.readframe, args=(url,))
        self.process_thread = threading.Thread(target=self.process_frames, args=(custom_frame_processor, rtmp_url))
        self.receive_thread.start()
        self.process_thread.start()
        print('VideoCapture started!')

    def release(self, custom_release=None):
        self.isstop = True
        if custom_release is not None:
            custom_release()
        print('VideoCapture stopped!')

    def readframe(self, url):
        capture = cv2.VideoCapture(url)
        while True:
            if capture.isOpened() or self.isstop:
                break
            print(f"Unable to open video source {url}. Retrying...")
            time.sleep(1)
            capture.release()
            capture = cv2.VideoCapture(url)
        self.size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.init_event.set()
        while not self.isstop:
            start = time.time()
            ok, frame = capture.read()
            if ok:
                self.q.put(frame)
            else:
                end = time.time()
                if end - start > 20:
                    while True:
                        print(f"Unable to open video source {url}. Retrying...")
                        time.sleep(1)
                        capture.release()
                        capture = cv2.VideoCapture(url)
                        if capture.isOpened() or self.isstop:
                            break
        capture.release()
        print("reading stopped!")

    def process_frames(self, frame_callback, rtmp_url):
        self.init_event.wait()
        (ffmpeg_process, ffmpeg_command) = init_rtmp_stream(self.size[0], self.size[1], rtmp_url)
        while not self.isstop or not self.q.empty():
            if self.q.empty():
                continue
            frame = self.q.get_nowait()
            frame = frame_callback(frame)
            if ffmpeg_process.poll() is not None:
                ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("Broken pipe error occurred!")
                time.sleep(1)
        ffmpeg_process.stdin.close()
        ffmpeg_process.communicate()
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
            cap = VideoCapture("rtmp://12.30.4.137:1935/live/aitest", "rtmp://12.30.4.137:1935/live/dsfdsfsfs",
                               frame_processor)
        elif "flightId" in data and len(data) == 1:
            if cap is not None:
                cap.release()


    client_handler = MQTTClientHandler(MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
                                       on_message_callback=on_message)
    asyncio.run(client_handler.run())
