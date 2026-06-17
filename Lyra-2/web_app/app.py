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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

def run_pipeline_thread(img_filename: str, prompt: str):
    global current_status, log_history
    
    stem = Path(img_filename).stem
    img_path = str(UPLOAD_DIR / img_filename)
    
    # 0. Cleanup old outputs
    output_video_path = WORKSPACE_ROOT / "outputs" / "zoomgs" / "videos" / f"{stem}.mp4"
    output_gs_dir = WORKSPACE_ROOT / "outputs" / "zoomgs" / "videos" / f"{stem}_gs_ours"
    
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
        
        cmd1 = [
            "python3", "-m", "lyra_2._src.inference.lyra2_zoomgs_inference",
            "--input_image_path", img_path,
            "--prompt", prompt,
            "--use_dmd",
            "--output_path", "outputs/zoomgs"
        ]
        
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
        
        while True:
            line = proc1.stdout.readline()
            if not line:
                break
            add_log(line)
            # Rough progress tracking
            if "Running DA3" in line:
                current_status["percent"] = 15
            elif "Generating ZOOM-IN" in line:
                current_status["percent"] = 25
            elif "Generating ZOOM-OUT" in line:
                current_status["percent"] = 35
                
        proc1.wait()
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
        if proc2.returncode != 0:
            raise RuntimeError(f"GS Reconstruction script exited with code {proc2.returncode}")
            
        add_log("[SYSTEM] Step 2 Complete! 3D GS Environment successfully generated!\n")
        current_status = {
            "stage": "completed",
            "percent": 100,
            "video_url": f"/outputs/zoomgs/videos/{stem}_gs_ours/gs_trajectory.mp4",
            "ply_url": f"/outputs/zoomgs/videos/{stem}_gs_ours/reconstructed_scene.ply"
        }
        
    except Exception as e:
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

@app.post("/api/run")
async def start_pipeline(filename: str = Form(...), prompt: str = Form(...)):
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
            run_pipeline_thread(filename, prompt)
            
    # Run in background thread
    threading.Thread(
        target=run_in_background,
        daemon=True
    ).start()
    
    return {"status": "started"}

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
