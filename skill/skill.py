import base64
import math
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import tf
from mcp.server.fastmcp import FastMCP

import rospy
from geometry_msgs.msg import PoseStamped
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_from_euler

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

NETWORK_INTERFACE = "eth0"
GROUNDINGDINO_SERVER = "http://pku4090.local:7579"
COLOR_IMAGE_TOPIC = "/camera/color/image_raw"
DEPTH_IMAGE_TOPIC = "/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"

mcp = FastMCP(name="GO2", stateless_http=True, host="0.0.0.0", port=8002)

WORK_AREA = [5.1,-2.450]
BAR_COUNTER = [-2,-8.5]
NURSING_HOUSE = [2.5,-12]

# ──────────────────────────────────────────────────────────────
# Open search points
# 开放式搜索巡逻点，均为 map 坐标系下的 [x, y]
# ──────────────────────────────────────────────────────────────

WORK_AREA_SEARCH_POINTS = [
    [-0.409, 1.429],
    [0.140, 4.797],
    [-5.348, 6.822],
    [-6.642, 1.437],
    [-1.122, 3.397],
    [2.396, 3.112],
]

NURSING_HOUSE_SEARCH_POINTS = [
    [1.784, 3.764],
    [2.559, 7.742],
    [0.004, 8.471],
    [1.474, 10.434],
    [3.186, 12.040],
    [0.489, 12.822],
    [2.037, 12.251],
]

BAR_COUNTER_SEARCH_POINTS = [
    [0.451, 6.014],
    [4.462, 6.400],
    [3.599, 9.117],
    [5.041, 9.704],
    [6.552, 8.453],
    [7.193, 5.712],
    [4.995, 4.232],
]


OPEN_SEARCH_POINTS = {
    "work_area": WORK_AREA_SEARCH_POINTS,
    "nursing_house": NURSING_HOUSE_SEARCH_POINTS,
    "bar_counter": BAR_COUNTER_SEARCH_POINTS,
}



def init_ros1():
    rospy.init_node("go2_skill_node", anonymous=True)


def ros_image_to_bgr(msg):
    if msg.encoding not in ("bgr8", "rgb8"):
        raise RuntimeError(f"Unsupported color image encoding: {msg.encoding}")

    channels = 3
    row_width = msg.width * channels
    data = np.frombuffer(msg.data, dtype=np.uint8)
    image = data.reshape((msg.height, msg.step))[:, :row_width]
    image = image.reshape((msg.height, msg.width, channels)).copy()

    if msg.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def ros_image_to_depth_m(msg):
    if msg.encoding == "32FC1":
        dtype = np.float32
        scale = 1.0
    elif msg.encoding == "16UC1":
        dtype = np.uint16
        scale = 0.001
    else:
        raise RuntimeError(f"Unsupported depth image encoding: {msg.encoding}")

    item_size = np.dtype(dtype).itemsize
    row_width = msg.width * item_size
    data = np.frombuffer(msg.data, dtype=np.uint8)
    rows = data.reshape((msg.height, msg.step))[:, :row_width]
    depth = rows.reshape((msg.height, msg.width, item_size)).copy().view(dtype)
    return depth.reshape((msg.height, msg.width)).astype(np.float32) * scale


def capture_frame_from_ros_topics(
    color_topic=COLOR_IMAGE_TOPIC,
    depth_topic=DEPTH_IMAGE_TOPIC,
    camera_info_topic=CAMERA_INFO_TOPIC,
    timeout=3.0,
):
    color_msg = rospy.wait_for_message(color_topic, Image, timeout=timeout)
    depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=timeout)
    info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=timeout)

    return ros_image_to_bgr(color_msg), ros_image_to_depth_m(depth_msg), info_msg, {
        "color_topic": color_topic,
        "depth_topic": depth_topic,
        "camera_info_topic": camera_info_topic,
        "frame_id": color_msg.header.frame_id,
        "width": color_msg.width,
        "height": color_msg.height,
        "color_stamp": color_msg.header.stamp.to_sec(),
        "depth_stamp": depth_msg.header.stamp.to_sec(),
    }


def call_groundingdino_api(
    image_bgr,
    caption,
    server=GROUNDINGDINO_SERVER,
    box_threshold=0.2,
    text_threshold=0.2,
):
    ok, jpg = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame as jpg")

    files = {
        "image": ("frame.jpg", jpg.tobytes(), "image/jpeg")
    }
    data = {
        "caption": caption,
        "box_threshold": str(box_threshold),
        "text_threshold": str(text_threshold),
    }

    resp = requests.post(f"{server}/detect", files=files, data=data, timeout=10)
    resp.raise_for_status()

    result = resp.json()
    return result.get("detections", []), result


def depth_at_pixel(depth_image, cx, cy, window=5):
    distance_m = float(depth_image[cy, cx])
    if np.isfinite(distance_m) and distance_m > 0:
        return distance_m

    radius = window // 2
    y1 = max(0, cy - radius)
    y2 = min(depth_image.shape[0], cy + radius + 1)
    x1 = max(0, cx - radius)
    x2 = min(depth_image.shape[1], cx + radius + 1)
    patch = depth_image[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def detection_to_camera_point(det, depth_image, camera_info):
    bbox = det.get("bbox", None)
    if bbox is None or len(bbox) != 4:
        return None

    height, width = depth_image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width - 1, x2)))
    y2 = int(max(0, min(height - 1, y2)))

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    distance_m = depth_at_pixel(depth_image, cx, cy)
    if distance_m <= 0:
        return None

    fx = camera_info.K[0]
    fy = camera_info.K[4]
    ppx = camera_info.K[2]
    ppy = camera_info.K[5]
    point_x = (cx - ppx) / fx * distance_m
    point_y = (cy - ppy) / fy * distance_m
    point_z = distance_m

    return {
        "bbox": [x1, y1, x2, y2],
        "center_pixel": {
            "x": cx,
            "y": cy,
        },
        "distance_m": distance_m,
        "camera_point": {
            "x": point_x,
            "y": point_y,
            "z": point_z,
        },
    }


def camera_point_to_map_goal(camera_point, stop_distance_m=0.6):
    pose = get_current_position_and_orientation(use_euler=True)
    robot_x = pose["position"]["x"]
    robot_y = pose["position"]["y"]
    yaw = pose["orientation"]["yaw"]

    # RealSense optical frame: x right, y down, z forward.
    forward_m = camera_point["z"]
    left_m = -camera_point["x"]

    goal_forward_m = max(0.0, forward_m - stop_distance_m)
    goal_left_m = left_m

    goal_x = robot_x + goal_forward_m * math.cos(yaw) - goal_left_m * math.sin(yaw)
    goal_y = robot_y + goal_forward_m * math.sin(yaw) + goal_left_m * math.cos(yaw)
    goal_yaw = math.atan2(goal_y - robot_y, goal_x - robot_x)

    return {
        "frame": "map",
        "x": goal_x,
        "y": goal_y,
        "yaw": goal_yaw,
        "stop_distance_m": stop_distance_m,
        "robot_pose": pose,
    }

@mcp.tool()
def get_current_position_and_orientation(use_euler: bool = True) -> dict:
    """获取当前位置和姿态

    Args:
        use_euler: 是否将姿态转换为欧拉角，默认 True，返回欧拉角
    """
    msg = rospy.wait_for_message("/Odometry", PoseStamped)

    result = {
        "position": {
            "x": msg.pose.position.x,
            "y": msg.pose.position.y,
            "z": msg.pose.position.z,
        }
    }

    if use_euler:
        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)

        result["orientation"] = {
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
    else:
        result["orientation"] = {
            "x": msg.pose.orientation.x,
            "y": msg.pose.orientation.y,
            "z": msg.pose.orientation.z,
            "w": msg.pose.orientation.w,
        }

    return result


@mcp.tool()
def detect_nearest_goal(
    caption: str,
    stop_distance_m: float = 1.0,
    box_threshold: float = 0.4,
    text_threshold: float = 0.4,
    server: str = GROUNDINGDINO_SERVER,
) -> dict:
    """根据远程传入的 caption 检测最近目标，并直接发送导航 goal

    Args:
        caption: GroundingDINO 检测提示词，例如 "person. bottle. chair"
        stop_distance_m: 导航目标距离物体预留距离，单位米，默认 1.0
        box_threshold: GroundingDINO box threshold
        text_threshold: GroundingDINO text threshold
        server: GroundingDINO API server
    """
    if not caption or not caption.strip():
        return {
            "success": False,
            "message": "caption 不能为空",
        }

    try:
        color_image, depth_image, camera_info_msg, camera_info = capture_frame_from_ros_topics()
        detections, raw_result = call_groundingdino_api(
            color_image,
            caption.strip(),
            server=server,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        candidates = []
        for det in detections:
            point_info = detection_to_camera_point(det, depth_image, camera_info_msg)
            if point_info is None:
                continue

            candidate = {
                "phrase": det.get("phrase", ""),
                "confidence": det.get("confidence", 0.0),
                **point_info,
            }
            candidates.append(candidate)

        if not candidates:
            return {
                "success": False,
                "message": "没有检测到带有效深度的目标",
                "caption": caption,
                "detections_count": len(detections),
                "camera": camera_info,
                "inference_time_ms": raw_result.get("inference_time_ms", None),
            }

        nearest = min(candidates, key=lambda item: item["distance_m"])
        goal = camera_point_to_map_goal(
            nearest["camera_point"],
            stop_distance_m=stop_distance_m,
        )
        navigation = send_goal(goal["x"], goal["y"], goal["yaw"])

        return {
            "success": navigation.get("success", False),
            "message": "已找到最近目标并发送导航 goal",
            "caption": caption,
            "nearest": nearest,
            "goal": goal,
            "navigation": navigation,
            "candidates": candidates,
            "camera": camera_info,
            "inference_time_ms": raw_result.get("inference_time_ms", None),
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "caption": caption,
        }

@mcp.tool()
def send_goal(x, y, yaw):

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    rospy.loginfo("等待 move_base action server...")

    client.wait_for_server()

    rospy.loginfo("已连接 move_base")

    goal = MoveBaseGoal()

    # 目标点参考坐标系，通常用 map

    goal.target_pose.header.frame_id = "map"

    goal.target_pose.header.stamp = rospy.Time.now()

    # 目标位置

    goal.target_pose.pose.position.x = x

    goal.target_pose.pose.position.y = y

    goal.target_pose.pose.position.z = 0.0

    # yaw 转四元数

    q = quaternion_from_euler(0, 0, yaw)

    goal.target_pose.pose.orientation.x = q[0]

    goal.target_pose.pose.orientation.y = q[1]

    goal.target_pose.pose.orientation.z = q[2]

    goal.target_pose.pose.orientation.w = q[3]

    rospy.loginfo("发送目标点: x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw)

    client.send_goal(goal)

    finished = client.wait_for_result(rospy.Duration(120))

    if not finished:

        rospy.logwarn("导航超时，取消目标")

        client.cancel_goal()
        return {
            "success": False,
            "message": "导航超时，已取消目标",
            "target": {
                "x": x,
                "y": y,
                "yaw": yaw,
            },
        }

    else:

        state = client.get_state()

        rospy.loginfo("导航结束，状态码: %d", state)
        return {
            "success": state == 3,
            "message": "导航结束",
            "state": state,
            "target": {
                "x": x,
                "y": y,
                "yaw": yaw,
            },
        }

@mcp.tool()
def go_to_area(area: str, yaw: float = 0.0) -> dict:
    """前往指定区域

    Args:
        area: 区域名称，可选：
            - work_area
            - bar_counter
            - nursing_house
        yaw: 到达后的朝向，单位为弧度，默认 0.0
    """
    area_map = {
        "work_area": WORK_AREA,
        "bar_counter": BAR_COUNTER,
        "nursing_house": NURSING_HOUSE,
    }

    if area not in area_map:
        return {
            "success": False,
            "message": f"未知区域: {area}",
            "available_areas": list(area_map.keys()),
        }

    x, y = area_map[area]

    rospy.loginfo("准备前往区域 %s: x=%.2f, y=%.2f, yaw=%.2f", area, x, y, yaw)

    send_goal(x, y, yaw)

    return {
        "success": True,
        "message": f"已发送前往 {area} 的导航任务",
        "target": {
            "x": x,
            "y": y,
            "yaw": yaw,
        },
    }

@mcp.tool()
def open_search(area: str, yaw: float = 0.0) -> dict:
    """开放式搜索指定区域，按预设搜索点依次导航

    Args:
        area: 区域名称，可选：
            - work_area
            - bar_counter
            - nursing_house
        yaw: 每个搜索点到达后的朝向，单位为弧度，默认 0.0
    """
    if area not in OPEN_SEARCH_POINTS:
        return {
            "success": False,
            "message": f"未知搜索区域: {area}",
            "available_areas": list(OPEN_SEARCH_POINTS.keys()),
        }

    points = OPEN_SEARCH_POINTS[area]

    for idx, point in enumerate(points):
        x, y = point
        rospy.loginfo(
            "开放搜索 %s: 前往第 %d/%d 个点 x=%.3f, y=%.3f, yaw=%.3f",
            area,
            idx + 1,
            len(points),
            x,
            y,
            yaw,
        )
        send_goal(x, y, yaw)

    return {
        "success": True,
        "message": f"已完成 {area} 开放式搜索任务",
        "area": area,
        "points": points,
    }

# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_ros1()
    # send_goal(5.1,-2.450,0)  # 前往工作区
    mcp.run(transport="streamable-http")
