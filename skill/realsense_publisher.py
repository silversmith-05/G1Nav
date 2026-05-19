#!/usr/bin/env python3
import argparse

import numpy as np
import pyrealsense2 as rs
import rospy
from sensor_msgs.msg import CameraInfo, Image


def parse_args():
    parser = argparse.ArgumentParser(description="Publish RealSense RGB-D frames to ROS1 topics")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-id", default="camera_color_optical_frame")
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    return parser.parse_args(rospy.myargv()[1:])


def find_device():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found. Please connect the D435i.")

    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        if "435" in name or "D435" in name:
            return serial, name

    dev = devices[0]
    return dev.get_info(rs.camera_info.serial_number), dev.get_info(rs.camera_info.name)


def image_msg(data, encoding, frame_id, stamp):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = data.shape[0]
    msg.width = data.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = data.strides[0]
    msg.data = data.tobytes()
    return msg


def camera_info_msg(intrinsics, frame_id, stamp):
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = intrinsics.height
    msg.width = intrinsics.width
    msg.distortion_model = "plumb_bob"
    msg.D = list(intrinsics.coeffs)
    msg.K = [
        intrinsics.fx, 0.0, intrinsics.ppx,
        0.0, intrinsics.fy, intrinsics.ppy,
        0.0, 0.0, 1.0,
    ]
    msg.R = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    msg.P = [
        intrinsics.fx, 0.0, intrinsics.ppx, 0.0,
        0.0, intrinsics.fy, intrinsics.ppy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    return msg


def main():
    rospy.init_node("realsense_rgbd_publisher")
    args = parse_args()

    color_pub = rospy.Publisher(args.color_topic, Image, queue_size=1)
    depth_pub = rospy.Publisher(args.depth_topic, Image, queue_size=1)
    info_pub = rospy.Publisher(args.camera_info_topic, CameraInfo, queue_size=1)

    serial, name = find_device()
    rospy.loginfo("Using RealSense device: %s, serial=%s", name, serial)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            stamp = rospy.Time.now()
            color = np.asanyarray(color_frame.get_data())
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            color_pub.publish(image_msg(color, "bgr8", args.frame_id, stamp))
            depth_pub.publish(image_msg(depth_m, "32FC1", args.frame_id, stamp))
            info_pub.publish(camera_info_msg(intrinsics, args.frame_id, stamp))
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
