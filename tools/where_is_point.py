#!/usr/bin/env python3
import os
import argparse
import yaml
import cv2


class PgmMapViewer:
    def __init__(self, yaml_path: str):
        self.yaml_path = os.path.abspath(yaml_path)
        self.yaml_dir = os.path.dirname(self.yaml_path)

        self.map_info = self.load_yaml(self.yaml_path)
        self.image_path = self.resolve_image_path(self.map_info["image"])

        self.resolution = float(self.map_info["resolution"])
        self.origin_x = float(self.map_info["origin"][0])
        self.origin_y = float(self.map_info["origin"][1])

        self.img_gray = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.img_gray is None:
            raise FileNotFoundError(f"无法读取图像: {self.image_path}")

        self.height, self.width = self.img_gray.shape[:2]
        self.img_color = cv2.cvtColor(self.img_gray, cv2.COLOR_GRAY2BGR)
        self.display_img = self.img_color.copy()

        self.window_name = "PGM Map Viewer"

    def load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        required_keys = ["image", "resolution", "origin"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"yaml 中缺少字段: {key}")

        return data

    def resolve_image_path(self, image_field: str) -> str:
        if os.path.isabs(image_field):
            return image_field
        return os.path.join(self.yaml_dir, image_field)

    def pixel_to_map(self, px: int, py: int):
        map_x = self.origin_x + px * self.resolution
        map_y = self.origin_y + (self.height - 1 - py) * self.resolution
        return map_x, map_y

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            map_x, map_y = self.pixel_to_map(x, y)

            self.display_img = self.img_color.copy()
            text = f"Map: ({map_x:.3f}, {map_y:.3f})"

            cv2.rectangle(
                self.display_img,
                (10, 10),
                (360, 48),
                (255, 255, 255),
                thickness=-1
            )

            cv2.putText(
                self.display_img,
                text,
                (18, 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            print(f"\rMap coordinate: ({map_x:.3f}, {map_y:.3f})", end="", flush=True)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        print("按 ESC 或 q 退出。")

        while True:
            cv2.imshow(self.window_name, self.display_img)
            key = cv2.waitKey(20) & 0xFF

            if key == 27 or key == ord("q"):
                break

        cv2.destroyAllWindows()
        print()


def main():
    parser = argparse.ArgumentParser(description="显示 PGM 地图并查看鼠标所在地图坐标")
    parser.add_argument("yaml_file", help="地图 yaml 文件路径")
    args = parser.parse_args()

    viewer = PgmMapViewer(args.yaml_file)
    viewer.run()


if __name__ == "__main__":
    main()