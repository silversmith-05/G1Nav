#!/usr/bin/env python3
import argparse
import os
import signal
import sys

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_SDK_DIRS = [
    os.environ.get("ORBBEC_SDK_DIR", ""),
    os.path.join(SCRIPT_DIR, "pyorbbecsdk"),
    os.path.join(REPO_ROOT, "pyorbbecsdk"),
    "/home/unitree/gaomj/pyorbbecsdk",
]

for sdk_dir in DEFAULT_SDK_DIRS:
    if sdk_dir and os.path.isdir(sdk_dir):
        os.environ["LD_LIBRARY_PATH"] = sdk_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        sys.path.insert(0, sdk_dir)
        examples_dir = os.path.join(sdk_dir, "examples")
        if os.path.isdir(examples_dir):
            sys.path.insert(0, examples_dir)
        break

from pyorbbecsdk import (  # noqa: E402
    Config,
    OBAlignMode,
    OBError,
    OBFormat,
    OBSensorType,
    Pipeline,
)


MIN_DEPTH_MM = 20
MAX_DEPTH_MM = 10000

running = True


def signal_handler(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def frame_to_bgr_image(frame):
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    if color_format == OBFormat.RGB:
        rgb = data.reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return data.reshape((height, width, 3)).copy()
    if color_format == OBFormat.RGBA:
        rgba = data.reshape((height, width, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if color_format == OBFormat.BGRA:
        bgra = data.reshape((height, width, 4))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format in (OBFormat.YUYV, OBFormat.YUY2):
        yuyv = data.reshape((height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        uyvy = data.reshape((height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.I420:
        i420 = data.reshape((height * 3 // 2, width))
        return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
    if color_format == OBFormat.YV12:
        yv12 = data.reshape((height * 3 // 2, width))
        return cv2.cvtColor(yv12, cv2.COLOR_YUV2BGR_YV12)
    if color_format == OBFormat.NV12:
        nv12 = data.reshape((height * 3 // 2, width))
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        nv21 = data.reshape((height * 3 // 2, width))
        return cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV21)
    if color_format in (OBFormat.GRAY, OBFormat.Y8):
        gray = data.reshape((height, width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    rospy.logwarn_throttle(5.0, "Unsupported color format: %s", color_format)
    return None


def depth_frame_to_mm(depth_frame, min_depth_mm, max_depth_mm):
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = float(depth_frame.get_depth_scale())
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    depth_mm = depth_data.astype(np.float32) * scale
    depth_mm = np.where((depth_mm > min_depth_mm) & (depth_mm < max_depth_mm), depth_mm, 0)
    return depth_mm.astype(np.uint16)


def choose_color_profile(pipeline, width, height, fps):
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        return profile_list.get_video_stream_profile(width, height, OBFormat.RGB, fps)
    except OBError:
        try:
            return profile_list.get_video_stream_profile(width, 0, OBFormat.RGB, fps)
        except OBError:
            rospy.logwarn("Requested color profile unavailable, using default color profile.")
            return profile_list.get_default_video_stream_profile()


def choose_aligned_depth_profile(pipeline, color_profile, align_mode):
    profile_list = pipeline.get_d2c_depth_profile_list(color_profile, align_mode)
    color_w = color_profile.get_width()
    color_h = color_profile.get_height()
    color_fps = color_profile.get_fps()

    fallback = profile_list.get_default_video_stream_profile()
    same_size = None
    for i in range(profile_list.get_count()):
        profile = profile_list.get_stream_profile_by_index(i).as_video_stream_profile()
        if profile.get_width() == color_w and profile.get_height() == color_h:
            if same_size is None:
                same_size = profile
            if profile.get_fps() == color_fps:
                return profile
    return same_size or fallback


def image_msg_from_bgr(image, stamp, frame_id):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = image.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(image).tobytes()
    return msg


def image_msg_from_depth_mm(depth_mm, stamp, frame_id):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = depth_mm.shape[:2]
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 2
    msg.data = np.ascontiguousarray(depth_mm).tobytes()
    return msg


def camera_info_msg_from_profile(color_profile, stamp, frame_id):
    intrinsic = color_profile.get_intrinsic()
    info = CameraInfo()
    info.header.stamp = stamp
    info.header.frame_id = frame_id
    info.width = int(intrinsic.width)
    info.height = int(intrinsic.height)
    info.distortion_model = "plumb_bob"

    try:
        distortion = color_profile.get_distortion()
        info.D = [
            float(distortion.k1),
            float(distortion.k2),
            float(distortion.p1),
            float(distortion.p2),
            float(distortion.k3),
        ]
    except Exception:
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]

    fx = float(intrinsic.fx)
    fy = float(intrinsic.fy)
    cx = float(intrinsic.cx)
    cy = float(intrinsic.cy)
    info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def parse_args():
    parser = argparse.ArgumentParser(description="Publish Orbbec Gemini RGB and aligned depth to ROS1.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--frame-id", default="camera_color_optical_frame")
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--min-depth-mm", type=float, default=MIN_DEPTH_MM)
    parser.add_argument("--max-depth-mm", type=float, default=MAX_DEPTH_MM)
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("gemini_publisher", anonymous=False)

    color_pub = rospy.Publisher(args.color_topic, Image, queue_size=args.queue_size)
    depth_pub = rospy.Publisher(args.depth_topic, Image, queue_size=args.queue_size)
    info_pub = rospy.Publisher(args.camera_info_topic, CameraInfo, queue_size=args.queue_size)

    pipeline = Pipeline()
    config = Config()
    align_mode = OBAlignMode.SW_MODE

    color_profile = choose_color_profile(pipeline, args.width, args.height, args.fps)
    depth_profile = choose_aligned_depth_profile(pipeline, color_profile, align_mode)
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_align_mode(align_mode)

    device_info = pipeline.get_device().get_device_info()
    rospy.loginfo("Device: %s serial=%s firmware=%s",
                  device_info.get_name(),
                  device_info.get_serial_number(),
                  device_info.get_firmware_version())
    rospy.loginfo("Color: %dx%d @%dfps %s",
                  color_profile.get_width(),
                  color_profile.get_height(),
                  color_profile.get_fps(),
                  color_profile.get_format())
    rospy.loginfo("Depth aligned to color: %dx%d @%dfps %s",
                  depth_profile.get_width(),
                  depth_profile.get_height(),
                  depth_profile.get_fps(),
                  depth_profile.get_format())

    pipeline.start(config)
    rospy.loginfo("Publishing %s, %s, %s",
                  args.color_topic,
                  args.depth_topic,
                  args.camera_info_topic)

    try:
        for _ in range(max(0, args.warmup)):
            if rospy.is_shutdown() or not running:
                break
            pipeline.wait_for_frames(1000)

        while not rospy.is_shutdown() and running:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                rospy.logwarn_throttle(5.0, "Waiting for both color and depth frames.")
                continue
            if depth_frame.get_format() != OBFormat.Y16:
                rospy.logwarn_throttle(5.0, "Unsupported depth format: %s", depth_frame.get_format())
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue
            depth_mm = depth_frame_to_mm(depth_frame, args.min_depth_mm, args.max_depth_mm)

            stamp = rospy.Time.now()
            color_pub.publish(image_msg_from_bgr(color_image, stamp, args.frame_id))
            depth_pub.publish(image_msg_from_depth_mm(depth_mm, stamp, args.frame_id))
            info_pub.publish(camera_info_msg_from_profile(color_profile, stamp, args.frame_id))
    finally:
        pipeline.stop()
        rospy.loginfo("Gemini publisher stopped.")


if __name__ == "__main__":
    main()
