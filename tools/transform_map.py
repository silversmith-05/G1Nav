#!/usr/bin/env python3
import argparse
import math
import os
import struct
from typing import Dict, List, Tuple

import yaml


DEFAULT_MATRIX = [
    [-0.938191336, -0.346117057, -0.22],
    [0.346117057, -0.938191336, -0.08],
    [0.0, 0.0, 1.0],
]
DEFAULT_YAW = 2.78816348


class PcdMeta:
    def __init__(self, header_lines: List[bytes], data_type: str, fields: List[str],
                 sizes: List[int], types: List[str], counts: List[int], points: int):
        self.header_lines = header_lines
        self.data_type = data_type
        self.fields = fields
        self.sizes = sizes
        self.types = types
        self.counts = counts
        self.points = points
        self.field_offsets = self._build_field_offsets()
        self.point_step = sum(size * count for size, count in zip(sizes, counts))

    def _build_field_offsets(self) -> Dict[str, Tuple[int, int, str]]:
        offsets = {}
        offset = 0
        for field, size, typ, count in zip(self.fields, self.sizes, self.types, self.counts):
            offsets[field] = (offset, size, typ)
            offset += size * count
        return offsets


def parse_header(path: str) -> Tuple[PcdMeta, int]:
    header_lines = []
    parsed = {}
    data_offset = 0

    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PCD header is missing DATA line: {path}")
            header_lines.append(line)
            data_offset += len(line)
            text = line.decode("ascii", errors="replace").strip()
            if text and not text.startswith("#"):
                parts = text.split()
                parsed[parts[0].upper()] = parts[1:]
                if parts[0].upper() == "DATA":
                    break

    fields = parsed.get("FIELDS")
    sizes = [int(v) for v in parsed.get("SIZE", [])]
    types = parsed.get("TYPE")
    counts = [int(v) for v in parsed.get("COUNT", ["1"] * len(fields))]
    data_type = parsed.get("DATA", [""])[0]
    points = int(parsed.get("POINTS", parsed.get("WIDTH", ["0"]))[0])

    if not fields or not sizes or not types:
        raise ValueError(f"PCD header must contain FIELDS, SIZE and TYPE: {path}")
    if len(fields) != len(sizes) or len(fields) != len(types) or len(fields) != len(counts):
        raise ValueError(f"PCD field metadata lengths do not match: {path}")
    if "x" not in fields or "y" not in fields or "z" not in fields:
        raise ValueError(f"PCD file must contain x, y and z fields: {path}")

    return PcdMeta(header_lines, data_type, fields, sizes, types, counts, points), data_offset


def lzf_decompress(data: bytes, expected_size: int) -> bytes:
    out = bytearray()
    i = 0

    while i < len(data):
        ctrl = data[i]
        i += 1

        if ctrl < 32:
            length = ctrl + 1
            out.extend(data[i:i + length])
            i += length
        else:
            length = ctrl >> 5
            ref_offset = (ctrl & 0x1F) << 8
            if length == 7:
                length += data[i]
                i += 1
            ref_offset += data[i]
            i += 1

            ref = len(out) - ref_offset - 1
            length += 2
            if ref < 0:
                raise ValueError("Invalid LZF back reference")
            for _ in range(length):
                out.append(out[ref])
                ref += 1

    if len(out) != expected_size:
        raise ValueError(f"LZF output size mismatch: got {len(out)}, expected {expected_size}")
    return bytes(out)


def load_pcd_payload(path: str, meta: PcdMeta, data_offset: int) -> bytearray:
    with open(path, "rb") as f:
        f.seek(data_offset)
        if meta.data_type == "binary":
            payload = f.read(meta.point_step * meta.points)
            if len(payload) != meta.point_step * meta.points:
                raise ValueError(f"Binary PCD payload is shorter than expected: {path}")
            return bytearray(payload)

        if meta.data_type == "binary_compressed":
            sizes = f.read(8)
            if len(sizes) != 8:
                raise ValueError(f"Compressed PCD payload is missing sizes: {path}")
            compressed_size, uncompressed_size = struct.unpack("<II", sizes)
            compressed = f.read(compressed_size)
            if len(compressed) != compressed_size:
                raise ValueError(f"Compressed PCD payload is shorter than expected: {path}")
            payload = lzf_decompress(compressed, uncompressed_size)
            if len(payload) != meta.point_step * meta.points:
                raise ValueError(f"Compressed PCD uncompressed size does not match header: {path}")
            return compressed_pcd_to_aos(payload, meta)

    raise ValueError(f"Unsupported PCD DATA type '{meta.data_type}' in {path}")


def compressed_pcd_to_aos(payload: bytes, meta: PcdMeta) -> bytearray:
    """Convert PCL binary_compressed field-major payload to normal point-major bytes."""
    out = bytearray(meta.point_step * meta.points)
    src_offset = 0

    for field, size, _typ, count in zip(meta.fields, meta.sizes, meta.types, meta.counts):
        dst_offset = meta.field_offsets[field][0]
        field_step = size * count
        field_bytes = field_step * meta.points
        field_payload = payload[src_offset:src_offset + field_bytes]
        if len(field_payload) != field_bytes:
            raise ValueError(f"Compressed PCD field '{field}' is shorter than expected")

        for point_index in range(meta.points):
            src = point_index * field_step
            dst = point_index * meta.point_step + dst_offset
            out[dst:dst + field_step] = field_payload[src:src + field_step]

        src_offset += field_bytes

    if src_offset != len(payload):
        raise ValueError("Compressed PCD payload has trailing bytes after field conversion")
    return out


def scalar_format(size: int, typ: str) -> str:
    key = (typ.upper(), size)
    formats = {
        ("F", 4): "f",
        ("F", 8): "d",
        ("I", 1): "b",
        ("I", 2): "h",
        ("I", 4): "i",
        ("I", 8): "q",
        ("U", 1): "B",
        ("U", 2): "H",
        ("U", 4): "I",
        ("U", 8): "Q",
    }
    if key not in formats:
        raise ValueError(f"Unsupported PCD scalar type: TYPE={typ}, SIZE={size}")
    return "<" + formats[key]


def read_scalar(buf: bytearray, base: int, meta: PcdMeta, field: str):
    offset, size, typ = meta.field_offsets[field]
    return struct.unpack_from(scalar_format(size, typ), buf, base + offset)[0]


def write_scalar(buf: bytearray, base: int, meta: PcdMeta, field: str, value):
    offset, size, typ = meta.field_offsets[field]
    struct.pack_into(scalar_format(size, typ), buf, base + offset, value)


def transform_xy(x: float, y: float, matrix: List[List[float]]) -> Tuple[float, float]:
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
    )


def transform_pcd(input_path: str, output_path: str, matrix: List[List[float]]):
    meta, data_offset = parse_header(input_path)

    if meta.data_type == "ascii":
        transform_ascii_pcd(input_path, output_path, meta, matrix)
        return

    payload = load_pcd_payload(input_path, meta, data_offset)
    has_normals = all(field in meta.field_offsets for field in ["normal_x", "normal_y", "normal_z"])

    for point_index in range(meta.points):
        base = point_index * meta.point_step
        x = read_scalar(payload, base, meta, "x")
        y = read_scalar(payload, base, meta, "y")
        x_new, y_new = transform_xy(x, y, matrix)
        write_scalar(payload, base, meta, "x", x_new)
        write_scalar(payload, base, meta, "y", y_new)

        if has_normals:
            nx = read_scalar(payload, base, meta, "normal_x")
            ny = read_scalar(payload, base, meta, "normal_y")
            nx_new = matrix[0][0] * nx + matrix[0][1] * ny
            ny_new = matrix[1][0] * nx + matrix[1][1] * ny
            write_scalar(payload, base, meta, "normal_x", nx_new)
            write_scalar(payload, base, meta, "normal_y", ny_new)

    write_binary_pcd(output_path, meta, payload)


def transform_ascii_pcd(input_path: str, output_path: str, meta: PcdMeta, matrix: List[List[float]]):
    x_index = meta.fields.index("x")
    y_index = meta.fields.index("y")
    normal_x_index = meta.fields.index("normal_x") if "normal_x" in meta.fields else None
    normal_y_index = meta.fields.index("normal_y") if "normal_y" in meta.fields else None

    with open(input_path, "r", encoding="ascii", errors="replace") as src, \
            open(output_path, "w", encoding="ascii") as dst:
        in_data = False
        for line in src:
            if not in_data:
                dst.write(line)
                if line.strip().upper().startswith("DATA"):
                    in_data = True
                continue

            parts = line.split()
            if not parts:
                dst.write(line)
                continue

            x = float(parts[x_index])
            y = float(parts[y_index])
            x_new, y_new = transform_xy(x, y, matrix)
            parts[x_index] = f"{x_new:.9g}"
            parts[y_index] = f"{y_new:.9g}"

            if normal_x_index is not None and normal_y_index is not None:
                nx = float(parts[normal_x_index])
                ny = float(parts[normal_y_index])
                parts[normal_x_index] = f"{matrix[0][0] * nx + matrix[0][1] * ny:.9g}"
                parts[normal_y_index] = f"{matrix[1][0] * nx + matrix[1][1] * ny:.9g}"

            dst.write(" ".join(parts) + "\n")


def write_binary_pcd(output_path: str, meta: PcdMeta, payload: bytearray):
    with open(output_path, "wb") as f:
        for line in meta.header_lines:
            text = line.decode("ascii", errors="replace").strip()
            if text.upper().startswith("DATA"):
                f.write(b"DATA binary\n")
            else:
                f.write(line)
        f.write(payload)


def transform_yaml(input_path: str, output_path: str, matrix: List[List[float]], yaw_delta: float):
    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "origin" not in data or len(data["origin"]) < 3:
        raise ValueError(f"Map yaml must contain origin: [x, y, yaw]: {input_path}")

    old_x, old_y, old_yaw = [float(v) for v in data["origin"][:3]]
    new_x, new_y = transform_xy(old_x, old_y, matrix)
    data["origin"] = [round(new_x, 6), round(new_y, 6), round(old_yaw + yaw_delta, 9)]

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def parse_matrix(values: List[float]) -> List[List[float]]:
    if len(values) != 9:
        raise ValueError("--matrix must contain 9 numbers")
    return [values[0:3], values[3:6], values[6:9]]


def default_output_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Transform ROS map yaml and PCD files from go2_map coordinates to g1_map coordinates."
    )
    parser.add_argument("--yaml", required=True, help="Input map.yaml in the source coordinate frame")
    parser.add_argument("--map-pcd", required=True, help="Input map.pcd in the source coordinate frame")
    parser.add_argument("--ground-pcd", required=True, help="Input ground_map.pcd in the source coordinate frame")
    parser.add_argument("--out-yaml", help="Output map yaml path. Default: input name with _g1 suffix")
    parser.add_argument("--out-map-pcd", help="Output transformed map pcd path. Default: input name with _g1 suffix")
    parser.add_argument("--out-ground-pcd", help="Output transformed ground pcd path. Default: input name with _g1 suffix")
    parser.add_argument(
        "--matrix",
        nargs=9,
        type=float,
        default=[value for row in DEFAULT_MATRIX for value in row],
        metavar=("M00", "M01", "M02", "M10", "M11", "M12", "M20", "M21", "M22"),
        help="3x3 homogeneous transform matrix from source map to target map. Defaults to go2_map -> g1_map.",
    )
    parser.add_argument(
        "--yaw-rad",
        type=float,
        default=DEFAULT_YAW,
        help="Yaw delta in radians for map.yaml origin. Defaults to 2.78816348.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing files")
    args = parser.parse_args()

    matrix = parse_matrix(args.matrix)
    out_yaml = args.out_yaml or default_output_path(args.yaml, "_g1")
    out_map_pcd = args.out_map_pcd or default_output_path(args.map_pcd, "_g1")
    out_ground_pcd = args.out_ground_pcd or default_output_path(args.ground_pcd, "_g1")

    print("Transform matrix:")
    for row in matrix:
        print(f"  {row}")
    print(f"Yaw delta: {args.yaw_rad:.9f} rad ({math.degrees(args.yaw_rad):.6f} deg)")
    print(f"YAML:       {args.yaml} -> {out_yaml}")
    print(f"map PCD:    {args.map_pcd} -> {out_map_pcd}")
    print(f"ground PCD: {args.ground_pcd} -> {out_ground_pcd}")

    if args.dry_run:
        return

    transform_yaml(args.yaml, out_yaml, matrix, args.yaw_rad)
    transform_pcd(args.map_pcd, out_map_pcd, matrix)
    transform_pcd(args.ground_pcd, out_ground_pcd, matrix)
    print("Done.")


if __name__ == "__main__":
    main()
