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

def generate_haunted_house_trajectory(num_frames=481):
    """
    Generates a path that:
    1. Moves forward into the house (Phase 1: 0% to 20%)
    2. Regard à droite (Phase 2: 20% to 40%)
    3. Regard à gauche (Phase 3: 40% to 60%)
    4. Revenir devant (Phase 4: 60% to 75%)
    5. Entrer dans la maison et voir le salon/escalier (Phase 5: 75% to 100%)
    """
    w2c_matrices = []
    
    # Base intrinsics (assuming 1280x720 reference)
    intrinsics = np.array([
        [821.64, 0.0, 640.0],
        [0.0, 821.75, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    intrinsics_list = []

    # Frame boundaries
    b1 = int(round(num_frames * 0.25))
    b2 = int(round(num_frames * 0.45))
    b3 = int(round(num_frames * 0.65))
    b4 = int(round(num_frames * 0.80))

    # Camera positions and rotations in the world (c2w)
    for i in range(num_frames):
        # Initialize defaults
        pos = np.array([0.0, 0.0, 0.0])
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        
        if i <= b1:
            # Phase 1: Avancer vers la maison (Z: 0.0 -> 1.5)
            progress = i / b1 if b1 > 0 else 1.0
            pos[2] = 1.5 * progress
            
        elif i <= b2:
            # Phase 2: Regarder à droite (Z: 1.5 -> 2.0, Yaw: 0.0 -> 35.0)
            progress = (i - b1) / (b2 - b1) if (b2 - b1) > 0 else 1.0
            pos[2] = 1.5 + 0.5 * progress
            yaw = 35.0 * progress
            
        elif i <= b3:
            # Phase 3: Regarder à gauche (Z: 2.0 -> 2.5, Yaw: 35.0 -> -35.0)
            progress = (i - b2) / (b3 - b2) if (b3 - b2) > 0 else 1.0
            pos[2] = 2.0 + 0.5 * progress
            yaw = 35.0 - 70.0 * progress
            
        elif i <= b4:
            # Phase 4: Revenir devant (Z: 2.5 -> 3.0, Yaw: -35.0 -> 0.0)
            progress = (i - b3) / (b4 - b3) if (b4 - b3) > 0 else 1.0
            pos[2] = 2.5 + 0.5 * progress
            yaw = -35.0 + 35.0 * progress
            
        else:
            # Phase 5: Entrer dans la maison (Z: 3.0 -> 5.5, X: 0.0 -> 0.4, Y: 0.0 -> 0.25, Yaw: 0.0 -> 25.0, Pitch: 0.0 -> -15.0)
            progress = (i - b4) / (num_frames - 1 - b4) if (num_frames - 1 - b4) > 0 else 1.0
            pos[2] = 3.0 + 2.5 * progress
            pos[0] = 0.4 * progress
            pos[1] = 0.25 * progress
            yaw = 25.0 * progress
            pitch = -15.0 * progress

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
    # Default is 321 for final run
    num_frames = 321
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
    b1 = int(round(num_frames * 0.25))
    b2 = int(round(num_frames * 0.45))
    b3 = int(round(num_frames * 0.65))
    b4 = int(round(num_frames * 0.80))
    
    captions = {
        "0": "We advance towards the scary haunted house, getting closer to its old wooden porch.",
        str(b1 + 1): "We look to the right of the house, scanning the overgrown garden and decayed facade.",
        str(b2 + 1): "We slowly turn the camera to the left, examining the weathered walls and balconies.",
        str(b3 + 1): "We turn the camera back to look straight ahead at the front door.",
        str(b4 + 1): "We enter the house through the doorway, showing the dusty living room, the dining area, and a creaky wooden staircase rising on the right."
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
    print("  --num_frames 481 \\")
    print("  --output_path /workspace/lyra/outputs/custom_traj")

if __name__ == "__main__":
    main()
