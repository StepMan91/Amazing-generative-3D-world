#!/usr/bin/env python3
"""
Lyra 2.0 - Visualize reconstructed 3D Gaussian Splats & Camera Paths in Rerun
"""
import os
import sys
import argparse
import numpy as np
from plyfile import PlyData
import rerun as rr

def main():
    parser = argparse.ArgumentParser(description="Log Lyra 2.0 Gaussian Splats and cameras to Rerun Viewer")
    parser.add_argument("--ply", type=str, required=True, help="Path to reconstructed_scene.ply")
    parser.add_argument("--cameras", type=str, help="Path to cameras.npz")
    parser.add_argument("--rerun-ip", type=str, default="172.17.0.1", help="Rerun server IP address")
    args = parser.parse_args()

    if not os.path.exists(args.ply):
        print(f"[RERUN] Error: PLY file not found at {args.ply}", file=sys.stderr)
        sys.exit(1)

    print(f"[RERUN] Initializing Rerun stream connection to {args.rerun_ip}:9876...")
    rr.init("Lyra-2 Reconstruction", spawn=False)
    rr.connect_grpc(f"rerun+http://{args.rerun_ip}:9876/proxy")

    print(f"[RERUN] Reading PLY file: {args.ply}")
    try:
        plydata = PlyData.read(args.ply)
    except Exception as e:
        print(f"[RERUN] Error reading PLY file: {e}", file=sys.stderr)
        sys.exit(1)

    vertex = plydata['vertex']
    x = np.asarray(vertex['x'])
    y = np.asarray(vertex['y'])
    z = np.asarray(vertex['z'])
    positions = np.stack([x, y, z], axis=1)
    total_points = len(positions)
    print(f"[RERUN] Loaded point positions. Total vertices: {total_points}")

    # Opacity filtering (using sigmoid since splats are un-activated)
    print("[RERUN] Filtering points by opacity threshold...")
    opacity = np.asarray(vertex['opacity'])
    opacities = 1.0 / (1.0 + np.exp(-opacity))
    opacity_threshold = 0.1
    mask = opacities > opacity_threshold
    positions = positions[mask]
    print(f"[RERUN] Filtered out low-opacity points: {len(positions)} / {total_points} remaining")

    # Color activation from Spherical Harmonics (DC)
    print("[RERUN] Activating colors from spherical harmonics (DC coefficients)...")
    f_dc_0 = np.asarray(vertex['f_dc_0'])[mask]
    f_dc_1 = np.asarray(vertex['f_dc_1'])[mask]
    f_dc_2 = np.asarray(vertex['f_dc_2'])[mask]

    SH_C0 = 0.28209479177387814
    r = np.clip(0.5 + SH_C0 * f_dc_0, 0.0, 1.0)
    g = np.clip(0.5 + SH_C0 * f_dc_1, 0.0, 1.0)
    b = np.clip(0.5 + SH_C0 * f_dc_2, 0.0, 1.0)
    colors = np.stack([r, g, b], axis=1)

    # Downsample points if still too heavy for WebGL / Web Rerun Viewer
    MAX_POINTS = 500000
    if len(positions) > MAX_POINTS:
        step = len(positions) // MAX_POINTS
        positions = positions[::step]
        colors = colors[::step]
        print(f"[RERUN] Downsampled to {len(positions)} points (every {step}th point) for fluid browser performance")
    else:
        print(f"[RERUN] Using all {len(positions)} points")

    # Set coordinate space standard (RDF: Right-Down-Forward)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # Stream points to Rerun
    print("[RERUN] Streaming point cloud to Rerun server...")
    rr.log("world/points", rr.Points3D(positions=positions, colors=colors), static=True)

    # Process camera trajectory if cameras.npz exists
    if args.cameras and os.path.exists(args.cameras):
        print(f"[RERUN] Parsing camera trajectory from {args.cameras}...")
        try:
            cams = np.load(args.cameras)
            if 'w2c_render' in cams:
                w2c_render = cams['w2c_render']
                intrinsics = cams.get('intrinsics_vipe', cams.get('intrinsics_da3', None))
                
                print(f"[RERUN] Logging {len(w2c_render)} camera frames to timeline...")
                for i in range(len(w2c_render)):
                    # Log at specific frame index on timeline
                    rr.set_time("frame", sequence=i)

                    # Compute camera-to-world pose from world-to-camera matrix
                    w2c = w2c_render[i]
                    try:
                        c2w = np.linalg.inv(w2c)
                    except np.linalg.LinAlgError:
                        continue

                    # Extract translation and rotation
                    translation = c2w[:3, 3]
                    rotation_matrix = c2w[:3, :3]

                    # Log camera transform
                    rr.log(
                        "world/camera",
                        rr.Transform3D(
                            translation=translation,
                            mat3x3=rotation_matrix
                        )
                    )

                    # Log pinhole camera parameters
                    if intrinsics is not None and i < len(intrinsics):
                        K = intrinsics[i]
                        rr.log(
                            "world/camera",
                            rr.Pinhole(
                                image_from_camera=K,
                                width=1920,
                                height=1080
                            )
                        )
                print("[RERUN] Trajectory path logged successfully!")
            else:
                print("[RERUN] Warning: 'w2c_render' key not found in cameras.npz", file=sys.stderr)
        except Exception as e:
            print(f"[RERUN] Error parsing cameras.npz: {e}", file=sys.stderr)

    print("[RERUN] Finished streaming reconstruction outputs to Rerun!")

if __name__ == "__main__":
    main()
