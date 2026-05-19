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
from tf.transformations import quaternion_from_euler

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

NETWORK_INTERFACE = "eth0"

mcp = FastMCP(name="GO2", stateless_http=True, host="0.0.0.0", port=8002)

WORK_AREA = [5.1,-2.450]
BAR_COUNTER = [-2,-8.5]
NURSING_HOUSE = [2.5,-12]

# ──────────────────────────────────────────────────────────────
# Open search points
# 开放式搜索巡逻点，均为 map 坐标系下的 [x, y]
# ──────────────────────────────────────────────────────────────

WORK_AREA_SEARCH_POINTS = [
    [0.700, -1.350],
    [1.350, -4.700],
    [7.200, -4.700],
    [6.550, 0.800],
    [2.050, -2.950],
    [-1.350, -3.900],
]

NURSING_HOUSE_SEARCH_POINTS = [
    [-0.550, -4.300],
    [0.100, -8.300],
    [2.750, -8.100],
    [2.050, -10.450],
    [1.000, -12.550],
    [3.800, -12.350],
    [2.150, -12.350],
]

BAR_COUNTER_SEARCH_POINTS = [
    [1.480, -5.950],
    [-2.150, -7.700],
    [-0.400, -9.950],
    [-1.550, -11.000],
    [-3.400, -10.350],
    [-4.950, -8.000],
    [-3.400, -5.850],
]


OPEN_SEARCH_POINTS = {
    "work_area": WORK_AREA_SEARCH_POINTS,
    "nursing_house": NURSING_HOUSE_SEARCH_POINTS,
    "bar_counter": BAR_COUNTER_SEARCH_POINTS,
}



def init_ros1():
    rospy.init_node("go2_skill_node", anonymous=True)

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

    else:

        state = client.get_state()

        rospy.loginfo("导航结束，状态码: %d", state)

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