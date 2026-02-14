"""
CSV to MimicKit PKL Converter

Converts G1 robot motion data from CSV format to MimicKit's PKL format.

Usage:
    python csv_to_mimickit.py --input data/csv/walk1_subject1.csv --output data/motions/g1/g1_walk1.pkl
"""

import argparse
import numpy as np
import pickle


def convert_csv_to_pkl(csv_path, pkl_path, fps=30, loop_mode=1):
    """
    Convert CSV motion file to MimicKit PKL format.

    Args:
        csv_path: Path to input CSV file
        pkl_path: Path to output PKL file
        fps: Frames per second (default: 30)
        loop_mode: 0 = CLAMP, 1 = WRAP (default: 1)
    """
    # Read CSV file
    frames = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split by comma
            values = line.split(',')

            # CSV has 36 values: root(7) + 29 joints = 36
            # Root: X(0), Y(1), Z(2), QX(3), QY(4), QZ(5), QW(6)
            # Joints: 29 values at indices 7-35
            # PKL needs 35 values: root(6, skip Y) + 29 joints = 35
            values = [float(v) for v in values]  # 36 values

            # Remove root Y (index 1): keep X(0), Z-QW(2-6), joints(7-35)
            joint_data = values[0:1] + values[2:7] + values[7:36]

            frames.append(joint_data)

    # Convert to numpy array
    frames = np.array(frames, dtype=np.float32)

    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames.shape}")

    # Save as PKL
    motion_data = {
        "loop_mode": loop_mode,
        "fps": fps,
        "frames": frames.tolist()
    }

    with open(pkl_path, 'wb') as f:
        pickle.dump(motion_data, f)

    print(f"Saved to: {pkl_path}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to MimicKit PKL format")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output PKL file path")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--loop", type=int, default=1, choices=[0, 1], help="Loop mode: 0=CLAMP, 1=WRAP (default: 1)")

    args = parser.parse_args()

    convert_csv_to_pkl(args.input, args.output, args.fps, args.loop)
