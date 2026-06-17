#!/usr/bin/env python3
"""
Lyra 2.0 Trajectory Generator Utility
Generateur de trajectoire camera pour Lyra 2.0
"""

import sys
import os
import json
import numpy as np

def create_camera_matrix(pos, yaw, pitch, roll):
    """
    Creates a Camera-to-World (c2w) matrix from translation and Euler rotations.
    """
    # Convert angles to radians
    y = np.radians(yaw)
    p = np.radians(pitch)
    r = np.radians(roll)

    # Rotation matrices
    R_yaw = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p), np.cos(p)]
    ])

    R_roll = np.array([
        [np.cos(r), -np.sin(r), 0],
        [np.sin(r), np.cos(r), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix: yaw -> pitch -> roll
    R = R_yaw @ R_pitch @ R_roll

    # 4x4 c2w matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = pos
    return c2w

def generate_haunted_house_trajectory(num_frames=241):
    """
    Generates a path that:
    1. Moves forward into the house (frames 0 to 80)
    2. Enters and pans right towards the living room (frames 80 to 160)
    3. Moves forward into the room and climbs slightly (frames 160 to 241)
    """
    w2c_matrices = []
    
    # Base intrinsics (assuming 1280x720 reference)
    intrinsics = np.array([
        [821.64, 0.0, 640.0],
        [0.0, 821.75, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    intrinsics_list = []

    # Camera positions and rotations in the world (c2w)
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        # Initialize defaults
        pos = np.array([0.0, 0.0, 0.0])
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        
        if i <= 80:
            # Stage 1: Move forward into the porch (Z goes from 0 to -0.6)
            progress = i / 80.0
            pos[2] = -0.6 * progress
            
        elif i <= 160:
            # Stage 2: Enter door and turn right (Yaw goes from 0 to 45 degrees, X shifts right)
            progress = (i - 80) / 80.0
            pos[2] = -0.6 - (0.4 * progress)       # keep moving forward (Z -> -1.0)
            pos[0] = 0.3 * progress                # slide slightly right (X -> 0.3)
            yaw = 45.0 * progress                  # turn 45 degrees right
            
        else:
            # Stage 3: Walk deeper into the room (X -> 0.6, Z -> -1.5) and tilt head up (Pitch)
            progress = (i - 160) / (num_frames - 160)
            pos[0] = 0.3 + (0.3 * progress)
            pos[2] = -1.0 - (0.5 * progress)
            pos[1] = 0.1 * progress                # move slightly up
            yaw = 45.0 + (5.0 * progress)          # adjust turn slightly
            pitch = -5.0 * progress                # tilt head up slightly (negative pitch)

        # Convert c2w to w2c (World-to-Camera)
        c2w = create_camera_matrix(pos, yaw, pitch, roll)
        w2c = np.linalg.inv(c2w)
        
        w2c_matrices.append(w2c)
        intrinsics_list.append(intrinsics)

    return np.array(w2c_matrices, dtype=np.float32), np.array(intrinsics_list, dtype=np.float32)

def main():
    print("=== Lyra 2.0 Trajectory Generator ===")
    
    # Create output directories if they don't exist
    output_dir = "/workspace/lyra/Lyra-2/assets/custom_trajectory_examples/haunted_house"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate matrices
    num_frames = 241
    w2c, intrinsics = generate_haunted_house_trajectory(num_frames)
    
    # Save .npz file
    npz_path = os.path.join(output_dir, "trajectory.npz")
    np.savez(
        npz_path,
        w2c=w2c,
        intrinsics=intrinsics,
        image_height=720,
        image_width=1280
    )
    print(f"[OK] Saved camera trajectory to: {npz_path}")
    print(f"     Total frames: {num_frames}")
    print(f"     Coordinates: X (Left/Right), Y (Up/Down), Z (Forward/Backward)")
    print(f"     Starts at (0,0,0) and moves to {w2c[-1, :3, 3]} (w2c space)")

    # Save a template captions.json
    captions = {
        "0": "A camera slowly glides forward approaching the dilapidated front porch of the abandoned wooden house.",
        "81": "We pass through the broken doorway into the dark entrance hall, panning slowly to the right.",
        "161": "Inside the dusty room, we pan towards the ruined living room and kitchen filled with cobwebs and decaying furniture."
    }
    
    json_path = os.path.join(output_dir, "captions.json")
    with open(json_path, "w") as f:
        json.dump(captions, f, indent=4)
    print(f"[OK] Saved template captions file to: {json_path}")
    print("\nTo run this custom trajectory with your image:")
    print("docker exec -it lyra2_dev python3 -m lyra_2._src.inference.lyra2_custom_traj_inference \\")
    print("  --input_image_path /workspace/lyra/Lyra-2/assets/samples/your_house_image.png \\")
    print("  --trajectory_path /workspace/lyra/Lyra-2/assets/custom_trajectory_examples/haunted_house/trajectory.npz \\")
    print("  --captions_path /workspace/lyra/Lyra-2/assets/custom_trajectory_examples/haunted_house/captions.json \\")
    print("  --num_frames 241 \\")
    print("  --output_path /workspace/lyra/outputs/custom_traj")

if __name__ == "__main__":
    main()
