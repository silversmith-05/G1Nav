#!/usr/bin/env python3
import argparse
import time
import threading

import cv2
import numpy as np
import requests
import pyrealsense2 as rs


def parse_args():
    parser = argparse.ArgumentParser(
        description="RealSense D435i + GroundingDINO realtime detection"
    )
    parser.add_argument("--server", default="http://pku4090.local:7579",
                        help="GroundingDINO API server, e.g. http://192.168.1.100:7579")
    parser.add_argument("--caption", default="phone. mouse . box",
                        help='GroundingDINO prompt, e.g. "car. person. dog"')
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--detect-interval", type=float, default=0.3,
                        help="Detection interval in seconds. Smaller = more frequent but slower.")
    parser.add_argument("--show-depth", action="store_true",
                        help="Show depth image next to color image.")
    return parser.parse_args()


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


def make_pipeline(args, serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return pipeline, align, depth_scale


def call_groundingdino_api(image_bgr, args):
    """
    image_bgr: OpenCV BGR image from RealSense
    return: detections list
    """
    ok, jpg = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame as jpg")

    files = {
        "image": ("frame.jpg", jpg.tobytes(), "image/jpeg")
    }

    data = {
        "caption": args.caption,
        "box_threshold": str(args.box_threshold),
        "text_threshold": str(args.text_threshold),
    }

    url = f"{args.server}/detect"
    resp = requests.post(url, files=files, data=data, timeout=10)
    resp.raise_for_status()

    result = resp.json()
    return result.get("detections", []), result


def draw_detections(image, detections, depth_frame=None, color_intrinsics=None):
    """
    Draw GroundingDINO bbox on image.
    bbox format: [x1, y1, x2, y2]
    """
    h, w = image.shape[:2]

    for det in detections:
        bbox = det.get("bbox", None)
        phrase = det.get("phrase", "")
        conf = det.get("confidence", 0.0)

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        position_text = ""
        if depth_frame is not None:
            distance_m = depth_frame.get_distance(cx, cy)
            if distance_m > 0:
                if color_intrinsics is not None:
                    point = rs.rs2_deproject_pixel_to_point(
                        color_intrinsics,
                        [cx, cy],
                        distance_m,
                    )
                    position_text = (
                        f" ({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})m"
                    )
                else:
                    position_text = f" {distance_m:.2f}m"
            else:
                position_text = " (--,--,--)m"

        label = f"{phrase} {conf:.2f}{position_text}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)

        text_y = max(20, y1 - 8)
        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main():
    args = parse_args()

    serial, name = find_device()
    print(f"Using RealSense device: {name}, serial={serial}")
    print(f"GroundingDINO server: {args.server}")
    print(f"Caption: {args.caption!r}")
    print("Press q or Esc to quit.")

    pipeline, align, depth_scale = make_pipeline(args, serial)

    latest_detections = []
    latest_result = None
    latest_error = None
    detecting = False
    last_detect_time = 0.0
    lock = threading.Lock()

    def detect_worker(frame_for_detection):
        nonlocal latest_detections, latest_result, latest_error, detecting

        try:
            detections, result = call_groundingdino_api(frame_for_detection, args)
            with lock:
                latest_detections = detections
                latest_result = result
                latest_error = None
        except Exception as e:
            with lock:
                latest_error = str(e)
        finally:
            with lock:
                detecting = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            now = time.time()

            # 按间隔启动一次检测线程，避免主画面完全卡住
            with lock:
                can_start_detect = not detecting

            if can_start_detect and now - last_detect_time >= args.detect_interval:
                frame_copy = color_image.copy()
                with lock:
                    detecting = True
                last_detect_time = now

                t = threading.Thread(
                    target=detect_worker,
                    args=(frame_copy,),
                    daemon=True,
                )
                t.start()

            display = color_image.copy()

            with lock:
                detections_copy = list(latest_detections)
                result_copy = latest_result
                error_copy = latest_error
                is_detecting = detecting

            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            draw_detections(display, detections_copy, depth_frame, color_intrinsics)

            # 中心深度显示
            center_x = depth_frame.get_width() // 2
            center_y = depth_frame.get_height() // 2
            center_depth_m = depth_frame.get_distance(center_x, center_y)

            cv2.circle(display, (center_x, center_y), 4, (0, 255, 255), -1)

            info = f"depth: {center_depth_m:.3f} m | det: {len(detections_copy)}"
            if is_detecting:
                info += " | detecting..."
            cv2.putText(
                display,
                info,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if result_copy is not None:
                infer_ms = result_copy.get("inference_time_ms", None)
                if infer_ms is not None:
                    cv2.putText(
                        display,
                        f"inference: {infer_ms:.1f} ms",
                        (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if error_copy:
                cv2.putText(
                    display,
                    f"API error: {error_copy[:80]}",
                    (12, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if args.show_depth:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET,
                )
                view = np.hstack((display, depth_colormap))
            else:
                view = display

            cv2.namedWindow("RealSense + GroundingDINO", cv2.WINDOW_NORMAL)
            cv2.imshow("RealSense + GroundingDINO", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
