import os
import sys
import json
import queue
import shutil
import threading
import subprocess
from pathlib import Path
from typing import Optional

import psutil
import pynvml
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Lyra 2.0 Generative 3D Environment Dashboard")

# Paths
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path("/workspace/lyra/Lyra-2")
UPLOAD_DIR = WORKSPACE_ROOT / "outputs" / "web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount outputs and static assets
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(WORKSPACE_ROOT / "outputs")), name="outputs")

# Global state
log_history = []
current_status = {"stage": "idle", "percent": 0}
pipeline_lock = threading.Lock()
current_process = None
pipeline_cancelled = False

# Initialize pynvml
try:
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as e:
    print(f"Failed to initialize NVML: {e}")
    NVML_AVAILABLE = False
    nvml_handle = None

def get_gpu_memory_usage() -> int:
    """Gets total GPU memory used by container python processes."""
    # 1. Try nvmi-smi query
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if res.returncode == 0 and res.stdout.strip():
            total = 0
            for line in res.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) == 2:
                    try:
                        total += int(parts[1].strip())
                    except:
                        pass
            if total > 0:
                return total
    except:
        pass

    # 2. Parse nvidia-smi text output
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            total = 0
            for line in res.stdout.split("\n"):
                if "python" in line and "MiB" in line:
                    parts = line.split()
                    for p in parts:
                        if p.endswith("MiB"):
                            try:
                                total += int(p[:-3])
                            except:
                                pass
            return total
    except:
        pass

    return 0

def add_log(text: str):
    log_history.append(text)
    sys.stdout.write(text)
    sys.stdout.flush()

def run_pipeline_thread(img_filename: str, prompt: str, trajectory: str, use_dmd: bool, num_frames: int, guidance: float, pose_scale: float):
    global current_status, log_history, current_process, pipeline_cancelled
    pipeline_cancelled = False
    
    stem = Path(img_filename).stem
    img_path = str(UPLOAD_DIR / img_filename)
    
    # Define paths based on trajectory type
    if trajectory == "preset":
        output_video_path = WORKSPACE_ROOT / "outputs" / "zoomgs" / "videos" / f"{stem}.mp4"
        output_gs_dir = WORKSPACE_ROOT / "outputs" / "zoomgs" / "videos" / f"{stem}_gs_ours"
        video_url = f"/outputs/zoomgs/videos/{stem}_gs_ours/gs_trajectory.mp4"
        ply_url = f"/outputs/zoomgs/videos/{stem}_gs_ours/reconstructed_scene.ply"
    else:
        output_video_path = WORKSPACE_ROOT / "outputs" / "custom_traj" / f"{stem}.mp4"
        output_gs_dir = WORKSPACE_ROOT / "outputs" / "custom_traj" / f"{stem}_gs_ours"
        video_url = f"/outputs/custom_traj/{stem}_gs_ours/gs_trajectory.mp4"
        ply_url = f"/outputs/custom_traj/{stem}_gs_ours/reconstructed_scene.ply"
        
    # Ensure parent directory of output video exists
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 0. Cleanup old outputs
    if output_video_path.exists():
        try: os.remove(output_video_path)
        except Exception as e: add_log(f"[SYSTEM] Warn: failed to remove old video: {e}\n")
        
    if output_gs_dir.exists():
        try: shutil.rmtree(output_gs_dir)
        except Exception as e: add_log(f"[SYSTEM] Warn: failed to remove old GS dir: {e}\n")

    env = os.environ.copy()
    env["LD_PRELOAD"] = "/workspace/lyra/libcuda_fake.so"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        # 1. Run video generation
        add_log("[SYSTEM] Starting Step 1: Exploration Video Generation...\n")
        current_status = {"stage": "video_gen", "percent": 5}
        
        if trajectory == "preset":
            num_zoom_in = 81
            num_zoom_out = max(1, num_frames - 80)
            
            cmd1 = [
                "python3", "-m", "lyra_2._src.inference.lyra2_zoomgs_inference",
                "--input_image_path", img_path,
                "--prompt", prompt,
                "--output_path", "outputs/zoomgs",
                "--num_frames_zoom_in", str(num_zoom_in),
                "--num_frames_zoom_out", str(num_zoom_out),
                "--guidance", str(guidance),
                "--zoom_out_strength", str(pose_scale)
            ]
            if use_dmd:
                cmd1.append("--use_dmd")
        else:
            traj_path = WORKSPACE_ROOT / "assets" / "custom_trajectory_examples" / trajectory / "trajectory.npz"
            captions_path = WORKSPACE_ROOT / "assets" / "custom_trajectory_examples" / trajectory / "captions.json"
            
            cmd1 = [
                "python3", "-m", "lyra_2._src.inference.lyra2_custom_traj_inference",
                "--input_image_path", img_path,
                "--trajectory_path", str(traj_path),
                "--output_path", "outputs/custom_traj",
                "--num_frames", str(num_frames),
                "--guidance", str(guidance),
                "--pose_scale", str(pose_scale)
            ]
            if captions_path.exists():
                cmd1 += ["--captions_path", str(captions_path)]
            else:
                cmd1 += ["--prompt", prompt]
                
            if use_dmd:
                cmd1.append("--use_dmd")
        
        add_log(f"[SYSTEM] Command: {' '.join(cmd1)}\n")
        
        proc1 = subprocess.Popen(
            cmd1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(WORKSPACE_ROOT),
            env=env
        )
        current_process = proc1
        
        while True:
            line = proc1.stdout.readline()
            if not line:
                break
            add_log(line)
            # Rough progress tracking
            if "Running DA3" in line:
                current_status["percent"] = 15
            elif "Generating ZOOM-IN" in line or "Generating video" in line:
                current_status["percent"] = 25
            elif "Generating ZOOM-OUT" in line or "Sampling:" in line:
                current_status["percent"] = 35
                
        proc1.wait()
        current_process = None
        if proc1.returncode != 0:
            raise RuntimeError(f"Video generation script exited with code {proc1.returncode}")
            
        add_log("[SYSTEM] Step 1 Complete! Combined video generated.\n")
        current_status = {"stage": "video_gen_done", "percent": 50}
        
        # Verify video output
        if not output_video_path.exists():
            raise FileNotFoundError("Video output file was not found where expected.")

        # 2. Run GS Reconstruction
        add_log("[SYSTEM] Starting Step 2: 3D Gaussian Splatting Reconstruction...\n")
        current_status = {"stage": "gs_recon", "percent": 55}
        
        cmd2 = [
            "python3", "-m", "lyra_2._src.inference.vipe_da3_gs_recon",
            "--input_video_path", str(output_video_path)
        ]
        
        add_log(f"[SYSTEM] Command: {' '.join(cmd2)}\n")
        
        proc2 = subprocess.Popen(
            cmd2,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(WORKSPACE_ROOT),
            env=env
        )
        current_process = proc2
        
        while True:
            line = proc2.stdout.readline()
            if not line:
                break
            add_log(line)
            # Rough progress tracking
            if "SLAM Pass (1/2)" in line:
                current_status["percent"] = 65
            elif "SLAM Pass (2/2)" in line:
                current_status["percent"] = 75
            elif "Model Forward" in line:
                current_status["percent"] = 85
            elif "Rendering" in line:
                current_status["percent"] = 92
                
        proc2.wait()
        current_process = None
        if proc2.returncode != 0:
            raise RuntimeError(f"GS Reconstruction script exited with code {proc2.returncode}")
            
        add_log("[SYSTEM] Step 2 Complete! 3D GS Environment successfully generated!\n")
        current_status = {
            "stage": "completed",
            "percent": 100,
            "video_url": video_url,
            "ply_url": ply_url
        }
        
    except Exception as e:
        current_process = None
        if pipeline_cancelled:
            add_log("[SYSTEM] Generation pipeline cancelled by user.\n")
            current_status = {"stage": "failed", "percent": 100, "error": "Pipeline cancelled by user"}
        else:
            add_log(f"[SYSTEM] ERROR: Pipeline execution failed: {str(e)}\n")
            current_status = {"stage": "failed", "percent": 100, "error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = BASE_DIR / "templates" / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    return HTMLResponse(content="<h3>Index template not found</h3>", status_code=404)

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    # Clean filename to avoid directory traversal
    filename = Path(file.filename).name
    file_path = UPLOAD_DIR / filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": filename, "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {e}")

@app.get("/api/trajectories")
async def get_trajectories():
    traj_dir = WORKSPACE_ROOT / "assets" / "custom_trajectory_examples"
    trajectories = ["preset"]
    if traj_dir.exists():
        for item in traj_dir.iterdir():
            if item.is_dir() and (item / "trajectory.npz").exists():
                trajectories.append(item.name)
    return sorted(trajectories)

@app.post("/api/run")
async def start_pipeline(
    filename: str = Form(...),
    prompt: str = Form(...),
    trajectory: str = Form(...),
    use_dmd: bool = Form(...),
    num_frames: int = Form(...),
    guidance: float = Form(5.0),
    pose_scale: float = Form(1.1)
):
    global log_history, current_status
    
    if not (UPLOAD_DIR / filename).exists():
        raise HTTPException(status_code=400, detail="Uploaded file does not exist.")
        
    if pipeline_lock.locked():
        raise HTTPException(status_code=409, detail="A generation pipeline is already running.")
        
    # Reset log and status
    log_history.clear()
    current_status = {"stage": "initializing", "percent": 0}
    
    def run_in_background():
        with pipeline_lock:
            run_pipeline_thread(
                filename,
                prompt,
                trajectory,
                use_dmd,
                num_frames,
                guidance,
                pose_scale
            )
            
    # Run in background thread
    threading.Thread(
        target=run_in_background,
        daemon=True
    ).start()
    
    return {"status": "started"}


@app.post("/api/stop")
async def stop_pipeline():
    global current_process, pipeline_cancelled
    if current_process is not None:
        try:
            pipeline_cancelled = True
            current_process.terminate()
            try:
                current_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                current_process.kill()
            return {"status": "stopping"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to terminate process: {e}")
    return {"status": "idle"}


@app.post("/api/stream_to_rerun")
async def stream_to_rerun(ply_url: str = Form(...)):
    if not ply_url.startswith("/outputs/"):
        raise HTTPException(status_code=400, detail="Invalid PLY URL format.")
        
    rel_path = ply_url.replace("/outputs/", "", 1)
    ply_path = WORKSPACE_ROOT / "outputs" / rel_path
    
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="PLY file not found.")
        
    cameras_path = ply_path.parent / "cameras.npz"
    cameras_arg = []
    if cameras_path.exists():
        cameras_arg = ["--cameras", str(cameras_path)]
        
    # Find Rerun IP
    rerun_ip = "172.17.0.1" # Default docker host gateway IP
    try:
        res = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=2)
        for line in res.stdout.splitlines():
            if line.startswith("default"):
                rerun_ip = line.split()[2]
                break
    except:
        pass
        
    # Start the python script and capture its output to log history
    cmd = [
        "python3", str(BASE_DIR / "visualize_in_rerun.py"),
        "--ply", str(ply_path),
        "--rerun-ip", rerun_ip
    ] + cameras_arg
    
    add_log(f"[SYSTEM] Streaming scene to Rerun at {rerun_ip}:9876...\n")
    add_log(f"[SYSTEM] Command: {' '.join(cmd)}\n")
    
    def run_stream():
        env = os.environ.copy()
        env["LD_PRELOAD"] = "" # Disable cuda fake if it causes issues with rerun
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                add_log(line)
            proc.wait()
            if proc.returncode == 0:
                add_log("[SYSTEM] Successfully completed Rerun stream!\n")
            else:
                add_log(f"[SYSTEM] Rerun stream exited with error code {proc.returncode}\n")
        except Exception as e:
            add_log(f"[SYSTEM] ERROR: Failed to run Rerun streaming: {e}\n")
            
    # Run in background thread
    threading.Thread(target=run_stream, daemon=True).start()
    
    return {"status": "started", "rerun_ip": rerun_ip}


@app.get("/api/ply_files")
async def list_ply_files(response: Response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    ply_list = []
    
    # Scan zoomgs outputs
    zoomgs_dir = WORKSPACE_ROOT / "outputs" / "zoomgs"
    if zoomgs_dir.exists():
        for root, dirs, files in os.walk(str(zoomgs_dir)):
            for f in files:
                if f.endswith(".ply"):
                    full_path = Path(root) / f
                    try:
                        rel_path = full_path.relative_to(WORKSPACE_ROOT / "outputs")
                        url = f"/outputs/{rel_path}"
                        label = f"Preset: {full_path.parent.name.replace('_gs_ours', '')}"
                        ply_list.append({"label": label, "url": url})
                    except Exception:
                        pass

    # Scan custom_traj outputs
    custom_traj_dir = WORKSPACE_ROOT / "outputs" / "custom_traj"
    if custom_traj_dir.exists():
        for root, dirs, files in os.walk(str(custom_traj_dir)):
            for f in files:
                if f.endswith(".ply"):
                    full_path = Path(root) / f
                    try:
                        rel_path = full_path.relative_to(WORKSPACE_ROOT / "outputs")
                        url = f"/outputs/{rel_path}"
                        label = f"Custom: {full_path.parent.name.replace('_gs_ours', '')}"
                        ply_list.append({"label": label, "url": url})
                    except Exception:
                        pass
                        
    return sorted(ply_list, key=lambda x: x["label"])


@app.get("/api/progress")
async def progress_stream():
    import asyncio
    async def event_generator():
        client_idx = 0
        while True:
            if client_idx < len(log_history):
                lines = log_history[client_idx:]
                client_idx = len(log_history)
                yield f"data: {json.dumps({'logs': lines, 'status': current_status})}\n\n"
            else:
                yield f"data: {json.dumps({'status': current_status})}\n\n"
                
            if current_status["stage"] in ["completed", "failed"] and client_idx >= len(log_history):
                # Ensure client receives final state
                yield f"data: {json.dumps({'status': current_status})}\n\n"
                break
                
            await asyncio.sleep(0.5)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/metrics")
async def metrics_stream():
    import asyncio
    async def event_generator():
        while True:
            # Query GPU Util
            gpu_util = 0
            gpu_temp = 0
            if NVML_AVAILABLE:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu
                except:
                    pass
                try:
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    pass
                    
            # Query GPU VRAM
            gpu_mem = get_gpu_memory_usage()
            
            # Query CPU and RAM
            cpu_util = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_util = ram.percent
            
            data = {
                "gpu_util": gpu_util,
                "gpu_mem": gpu_mem,
                "gpu_temp": gpu_temp,
                "cpu_util": cpu_util,
                "ram_util": ram_util
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1.0)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
