from os import getenv

RTMP_OUTPUT = (
    getenv("RTMP_OUPUT")
    if getenv("RTMP_OUTPUT")
    else "rtmp://59.46.115.177:1935/live/aiOutTest"
)

RTMP_INPUT = (
    getenv("RTMP_INPUT")
    if getenv("RTMP_INPUT")
    else "rtmp://59.46.115.177:1935/live/aitest"
)

MQTT_HOST = getenv["MQTT_HOST"] if getenv("MQTT_HOST") else "59.46.115.177"
MQTT_PORT = getenv["MQTT_PORT"] if getenv("MQTT_PORT") else 42417
MQTT_USERNAME = getenv("MQTT_USER") if getenv("MQTT_USER") else "admin"
MQTT_PASSWORD = getenv("MQTT_PASS") if getenv("MQTT_PASS") else "public"

MODEL_PATH = "./models/yolov8.engine"
TRACKER_CONFIG = (
    getenv("TRACKER_CONFIG")
    if getenv("TRACKER_CONFIG")
    else "./cfg/bytetrack.yaml"
 )

# USE_TENSORRT: bool = getenv("USE_TENSORRT") if getenv("USE_TENSORRT") else False
# IS_DOCKER: bool = getenv("DOCKER") if getenv("DOCKER") else False
