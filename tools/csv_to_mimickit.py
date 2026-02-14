"""
CSV to MimicKit PKL Converter

Converts G1 robot motion data from CSV format to MimicKit's PKL format.

Usage:
    python csv_to_mimickit.py --input data/csv/walk1_subject1.csv --output data/motions/g1/g1_walk1.pkl
"""

import argparse
import numpy as np
import pickle


def quaternion_to_expmap(qx, qy, qz, qw):
    """
    Convert quaternion to exponential map.
    """
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm < 1e-10:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    if qw > 0.99999:
        return 0.0, 0.0, 0.0
    if qw < -0.99999:
        return np.pi * 2 * np.array([qx, qy, qz]) / (np.sqrt(qx*qx + qy*qy + qz*qz) + 1e-10)

    angle = 2.0 * np.arccos(np.clip(qw, -1.0, 1.0))
    sin_half_angle = np.sqrt(1.0 - qw*qw)
    if sin_half_angle < 1e-10:
        return 0.0, 0.0, 0.0

    axis = np.array([qx, qy, qz]) / sin_half_angle
    expmap = axis * angle
    return expmap[0], expmap[1], expmap[2]


def convert_csv_to_pkl(csv_path, pkl_path, fps=30, loop_mode=1):
    """
    Convert CSV motion file to MimicKit PKL format.
    """
    # Use 0 as Y (let physics engine calculate height from ground)
    # Reference files have Y close to 0, so we set it to 0
    base_height = 0

    # Process frames
    frames = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            values = line.split(',')
            values = [float(v) for v in values]

            root_x = values[0]
            root_y = values[1]
            root_z = values[2]
            qx, qy, qz, qw = values[3], values[4], values[5], values[6]

            # Convert quaternion to exponential map
            ex, ey, ez = quaternion_to_expmap(qx, qy, qz, qw)

            # Use original Y value
            root_y_relative = root_y

            joint_data = [root_x, root_y_relative, root_z, ex, ey, ez] + values[7:36]
            frames.append(joint_data)

    frames = np.array(frames, dtype=np.float32)
    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames.shape}")

    motion_data = {
        "loop_mode": loop_mode,
        "fps": fps,
        "frames": frames.tolist()
    }

    with open(pkl_path, 'wb') as f:
        pickle.dump(motion_data, f)

    print(f"Saved to: {pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to MimicKit PKL format")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output PKL file path")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--loop", type=int, default=1, choices=[0, 1], help="Loop mode: 0=CLAMP, 1=WRAP (default: 1)")

    args = parser.parse_args()
    convert_csv_to_pkl(args.input, args.output, args.fps, args.loop)
