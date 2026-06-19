import * as GaussianSplats3D from 'gaussian-splats-3d';

// State Variables
let uploadedFilename = "";
let progressEventSource = null;
let metricsEventSource = null;
let splatViewer = null;
let currentPlyUrl = "";

// DOM Elements
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const imagePreview = document.getElementById("image-preview");
const dropZoneContent = document.querySelector(".drop-zone-content");
const generateBtn = document.getElementById("generate-btn");
const stopBtn = document.getElementById("stop-btn");
const plyHistorySelect = document.getElementById("ply-history-select");
const refreshPlyBtn = document.getElementById("refresh-ply-btn");
const uploadPlyTrigger = document.getElementById("upload-ply-trigger");
const localPlyInput = document.getElementById("local-ply-input");
const promptInput = document.getElementById("prompt-input");
const progressBar = document.getElementById("progress-bar");
const progressPct = document.getElementById("progress-pct");
const logTerminal = document.getElementById("log-terminal");
const clearLogsBtn = document.getElementById("clear-logs-btn");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

// Pipeline Step Elements
const stepUpload = document.getElementById("step-upload");
const stepVideo = document.getElementById("step-video");
const stepRecon = document.getElementById("step-recon");

// Output Elements
const outputCard = document.getElementById("output-card");
const outputVideo = document.getElementById("output-video");
const downloadPlyBtn = document.getElementById("download-ply-btn");
const streamRerunBtn = document.getElementById("stream-rerun-btn");

// Viewer Loader Elements
const splatViewerContainer = document.getElementById("splat-viewer-container");
const splatLoading = document.getElementById("splat-loading");
const splatSpinner = document.getElementById("splat-spinner");
const splatLoadingText = document.getElementById("splat-loading-text");
const splatLoadingSubtext = document.getElementById("splat-loading-subtext");

// Resource Card Value Labels
const gpuUtilVal = document.getElementById("gpu-util-val");
const gpuMemVal = document.getElementById("gpu-mem-val");
const gpuTempVal = document.getElementById("gpu-temp-val");
const cpuUtilVal = document.getElementById("cpu-util-val");
const ramUtilVal = document.getElementById("ram-util-val");

// Config Input Elements
const trajectorySelect = document.getElementById("trajectory-select");
const dmdSelect = document.getElementById("dmd-select");
const framesSelect = document.getElementById("frames-select");
const guidanceInput = document.getElementById("guidance-input");
const scaleInput = document.getElementById("scale-input");
const advancedToggle = document.getElementById("advanced-toggle");
const advancedPanel = document.getElementById("advanced-panel");

// ---------------------------------------------------------------------------
// 1. Chart.js Configurations
// ---------------------------------------------------------------------------
const charts = {};

function createResourceChart(canvasId, label, color) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    
    // Gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, 70);
    gradient.addColorStop(0, color + "50"); // 30% opacity
    gradient.addColorStop(1, color + "00"); // 0% opacity
    
    return new Chart(ctx, {
        type: "line",
        data: {
            labels: Array(30).fill(""),
            datasets: [{
                label: label,
                data: Array(30).fill(0),
                borderColor: color,
                borderWidth: 1.5,
                backgroundColor: gradient,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: {
                    display: false,
                    min: 0,
                    max: 100
                }
            },
            animation: false
        }
    });
}

// Initialize charts
charts.gpu = createResourceChart("gpuChart", "GPU Util", "#6366f1");
charts.temp = createResourceChart("gpuTempChart", "GPU Temp", "#f97316");
charts.vram = createResourceChart("vramChart", "VRAM Util", "#a855f7");
charts.cpu = createResourceChart("cpuChart", "CPU Util", "#3b82f6");
charts.ram = createResourceChart("ramChart", "RAM Util", "#06b6d4");

// VRAM can scale above 100 depending on total memory. Let's make it autoscale.
charts.vram.options.scales.y.max = undefined;
charts.vram.update();

// Update Chart helper
function updateChartData(chart, newValue) {
    chart.data.datasets[0].data.push(newValue);
    chart.data.datasets[0].data.shift();
    chart.update("none");
}

// Connect to Metrics Stream
function connectMetricsStream() {
    if (metricsEventSource) {
        metricsEventSource.close();
    }
    
    metricsEventSource = new EventSource("/api/metrics");
    
    metricsEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Update Labels
        gpuUtilVal.innerText = `${data.gpu_util}%`;
        gpuMemVal.innerText = `${data.gpu_mem} MB`;
        if (gpuTempVal) gpuTempVal.innerText = `${data.gpu_temp}°C`;
        cpuUtilVal.innerText = `${data.cpu_util}%`;
        ramUtilVal.innerText = `${data.ram_util}%`;
        
        // Update Charts
        updateChartData(charts.gpu, data.gpu_util);
        if (charts.temp) updateChartData(charts.temp, data.gpu_temp);
        updateChartData(charts.cpu, data.cpu_util);
        updateChartData(charts.ram, data.ram_util);

        
        // Compute rough VRAM utilization percentage for plotting (128GB total memory or Blackwell capacity)
        // Blackwell GB10 has 24GB or 32GB VRAM. Let's assume max scale in chart is 32000 MB.
        const vramPct = Math.min(100, (data.gpu_mem / 32768) * 100);
        updateChartData(charts.vram, vramPct);
    };
    
    metricsEventSource.onerror = () => {
        console.error("Metrics EventSource lost connection. Retrying...");
    };
}

// ---------------------------------------------------------------------------
// 2. Drag & Drop Upload
// ---------------------------------------------------------------------------
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

["dragleave", "dragend"].forEach(event => {
    dropZone.addEventListener(event, () => {
        dropZone.classList.remove("drag-over");
    });
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
        handleUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length) {
        handleUpload(fileInput.files[0]);
    }
});

async function handleUpload(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please upload an image file.");
        return;
    }
    
    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";
        dropZoneContent.style.display = "none";
    };
    reader.readAsDataURL(file);
    
    // Send to server
    appendTerminalLog("[SYSTEM] Uploading image to server...");
    updateStepStatus(stepUpload, "active");
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });
        
        if (res.ok) {
            const data = await res.json();
            uploadedFilename = data.filename;
            appendTerminalLog(`[SYSTEM] Image uploaded successfully as ${uploadedFilename}\n`);
            updateStepStatus(stepUpload, "completed");
            generateBtn.removeAttribute("disabled");
        } else {
            throw new Error(await res.text());
        }
    } catch (err) {
        appendTerminalLog(`[SYSTEM] ERROR: Failed to upload image: ${err.message}\n`);
        updateStepStatus(stepUpload, "failed");
    }
}

// ---------------------------------------------------------------------------
// 3. Execution Control & Log Terminal
// ---------------------------------------------------------------------------
generateBtn.addEventListener("click", async () => {
    if (!uploadedFilename) return;
    
    const prompt = promptInput.value.trim() || promptInput.placeholder;
    const trajectory = trajectorySelect ? trajectorySelect.value : "preset";
    const useDmd = dmdSelect ? dmdSelect.value === "true" : true;
    const numFrames = framesSelect ? parseInt(framesSelect.value, 10) : 161;
    const guidance = guidanceInput ? parseFloat(guidanceInput.value) : 5.0;
    const poseScale = scaleInput ? parseFloat(scaleInput.value) : 1.1;
    
    // Disable inputs
    disableAllInputs();
    
    // Clear logs and hide old outputs
    logTerminal.innerHTML = "";
    outputCard.style.display = "none";
    
    // Reset 3D viewer state
    currentPlyUrl = "";
    if (splatViewer) {
        try {
            splatViewer.dispose();
        } catch(e) {}
        splatViewer = null;
    }
    if (splatViewerContainer) {
        const canvases = splatViewerContainer.querySelectorAll("canvas");
        canvases.forEach(c => c.remove());
    }
    if (splatLoading) {
        if (splatSpinner) splatSpinner.style.display = "none";
        if (splatLoadingText) splatLoadingText.innerText = "Awaiting 3D Generation...";
        if (splatLoadingSubtext) splatLoadingSubtext.innerText = "The viewer will automatically display the reconstructed 3D environment once the pipeline finishes.";
        splatLoading.classList.remove("hidden");
    }
    
    appendTerminalLog("[SYSTEM] Initializing 3D Generative Pipeline...\n");
    
    // Reset steps UI
    updateStepStatus(stepVideo, "pending");
    updateStepStatus(stepRecon, "pending");
    
    // Trigger run
    const formData = new FormData();
    formData.append("filename", uploadedFilename);
    formData.append("prompt", prompt);
    formData.append("trajectory", trajectory);
    formData.append("use_dmd", useDmd);
    formData.append("num_frames", numFrames);
    formData.append("guidance", guidance);
    formData.append("pose_scale", poseScale);
    
    try {
        const res = await fetch("/api/run", {
            method: "POST",
            body: formData
        });
        
        if (res.ok) {
            connectProgressStream();
        } else {
            throw new Error(await res.text());
        }
    } catch (err) {
        appendTerminalLog(`[SYSTEM] ERROR: Failed to launch pipeline: ${err.message}\n`);
        enableAllInputs();
    }
});


function connectProgressStream() {
    if (progressEventSource) {
        progressEventSource.close();
    }
    
    progressEventSource = new EventSource("/api/progress");
    
    progressEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // 1. Append Logs
        if (data.logs) {
            data.logs.forEach(line => {
                appendTerminalLog(line);
            });
        }
        
        // 2. Update Status and Steps
        if (data.status) {
            const status = data.status;
            
            // Set Progress Bar
            progressBar.style.width = `${status.percent}%`;
            progressPct.innerText = `${status.percent}%`;
            
            // Update Status Banner
            updateStatusBanner(status.stage);
            
            // Step State Transitions
            if (status.stage === "video_gen") {
                updateStepStatus(stepUpload, "completed");
                updateStepStatus(stepVideo, "active");
            } else if (status.stage === "video_gen_done") {
                updateStepStatus(stepVideo, "completed");
            } else if (status.stage === "gs_recon") {
                updateStepStatus(stepVideo, "completed");
                updateStepStatus(stepRecon, "active");
            } else if (status.stage === "completed") {
                updateStepStatus(stepRecon, "completed");
                showFinalOutputs(status.video_url, status.ply_url);
                progressEventSource.close();
                enableAllInputs();
            } else if (status.stage === "failed") {
                // Find active step and mark failed
                if (stepVideo.classList.contains("active")) {
                    updateStepStatus(stepVideo, "failed");
                } else if (stepRecon.classList.contains("active")) {
                    updateStepStatus(stepRecon, "failed");
                }
                appendTerminalLog(`\n[SYSTEM] ERROR: Pipeline failed: ${status.error}\n`);
                progressEventSource.close();
                enableAllInputs();
            }
        }
    };
    
    progressEventSource.onerror = () => {
        console.error("Progress EventSource disconnected.");
    };
}

// Helper: Enable Inputs after run
function enableAllInputs() {
    generateBtn.style.display = "flex";
    generateBtn.removeAttribute("disabled");
    if (stopBtn) {
        stopBtn.style.display = "none";
        stopBtn.removeAttribute("disabled");
    }
    promptInput.removeAttribute("disabled");
    fileInput.removeAttribute("disabled");
    if (trajectorySelect) trajectorySelect.removeAttribute("disabled");
    if (dmdSelect) dmdSelect.removeAttribute("disabled");
    if (framesSelect) framesSelect.removeAttribute("disabled");
    if (guidanceInput) guidanceInput.removeAttribute("disabled");
    if (scaleInput) scaleInput.removeAttribute("disabled");
}

function disableAllInputs() {
    generateBtn.style.display = "none";
    if (stopBtn) {
        stopBtn.style.display = "flex";
        stopBtn.removeAttribute("disabled");
    }
    promptInput.setAttribute("disabled", "true");
    fileInput.setAttribute("disabled", "true");
    if (trajectorySelect) trajectorySelect.setAttribute("disabled", "true");
    if (dmdSelect) dmdSelect.setAttribute("disabled", "true");
    if (framesSelect) framesSelect.setAttribute("disabled", "true");
    if (guidanceInput) guidanceInput.setAttribute("disabled", "true");
    if (scaleInput) scaleInput.setAttribute("disabled", "true");
}


// Helper: Append logs to terminal
function appendTerminalLog(text) {
    const isScrollAtBottom = logTerminal.scrollHeight - logTerminal.clientHeight <= logTerminal.scrollTop + 5;
    
    const line = document.createElement("div");
    line.className = "terminal-line";
    if (text.startsWith("[SYSTEM]")) {
        line.classList.add("system-line");
    }
    line.innerText = text.trim();
    logTerminal.appendChild(line);
    
    if (isScrollAtBottom) {
        logTerminal.scrollTop = logTerminal.scrollHeight;
    }
}

// Helper: Update step visual class
function updateStepStatus(stepEl, statusClass) {
    stepEl.className = `step-item ${statusClass}`;
    const circle = stepEl.querySelector(".step-circle");
    if (statusClass === "completed") {
        circle.innerText = "✓";
    } else if (statusClass === "failed") {
        circle.innerText = "✗";
    } else if (statusClass === "active") {
        circle.innerText = "●";
    }
}

// Helper: Update Header Status Banner
function updateStatusBanner(stage) {
    if (stage === "idle" || stage === "completed") {
        statusDot.className = "status-dot green";
        statusText.innerText = "System Ready";
    } else if (stage === "failed") {
        statusDot.className = "status-dot orange";
        statusText.innerText = "System Error";
    } else {
        statusDot.className = "status-dot orange";
        statusText.innerText = `Running: ${stage.replace("_", " ").toUpperCase()}`;
    }
}

// Helper: Display Output Card
function showFinalOutputs(videoUrl, plyUrl) {
    outputVideo.src = videoUrl;
    currentPlyUrl = plyUrl;
    outputCard.style.display = "block";
    
    // Refresh history list so newly generated file is selected
    loadPlyHistory().then(() => {
        if (plyHistorySelect) {
            plyHistorySelect.value = plyUrl;
        }
    });
    
    // Load new splats in 3D viewer
    initSplatViewer(plyUrl);
    
    // Scroll output card into view
    outputCard.scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// 4. Interactive 3D Gaussian Splatting Viewer
// ---------------------------------------------------------------------------
function initSplatViewer(plyUrl) {
    if (!plyUrl) return;
    
    if (splatLoading) {
        if (splatSpinner) splatSpinner.style.display = "block";
        if (splatLoadingText) splatLoadingText.innerText = "Loading 3D Gaussian Splats...";
        if (splatLoadingSubtext) splatLoadingSubtext.innerText = "Controls: Left-Click to Orbit | Right-Click to Pan | WASD to Fly";
        splatLoading.classList.remove("hidden");
    }
    
    try {
        if (splatViewer) {
            splatViewer.dispose();
        }
    } catch(e) {
        console.error("Error disposing splatViewer:", e);
    }
    splatViewer = null;

    // Clean up old canvas elements AFTER disposing to prevent removeChild crash
    if (splatViewerContainer) {
        const canvases = splatViewerContainer.querySelectorAll("canvas");
        canvases.forEach(c => c.remove());
    }

    // Update download and rerun buttons
    if (downloadPlyBtn) {
        if (plyUrl.startsWith("blob:") || plyUrl.startsWith("data:")) {
            downloadPlyBtn.style.display = "none";
            if (streamRerunBtn) streamRerunBtn.style.display = "none";
        } else {
            downloadPlyBtn.href = plyUrl;
            downloadPlyBtn.style.display = "inline-flex";
            if (streamRerunBtn) {
                streamRerunBtn.style.display = "inline-flex";
                streamRerunBtn.removeAttribute("disabled");
                streamRerunBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 16px; height: 16px;">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                    <span>Stream to Rerun</span>
                `;
            }
        }
    }

    // Update Cinematic Replays & Walkthroughs Section
    const replaysContainer = document.getElementById("viewer-replays-container");
    const replayExplorationVideo = document.getElementById("replay-exploration-video");
    const replayGsVideo = document.getElementById("replay-gs-video");

    if (replaysContainer && replayExplorationVideo && replayGsVideo) {
        if (plyUrl.startsWith("blob:") || plyUrl.startsWith("data:")) {
            replaysContainer.style.display = "none";
        } else {
            try {
                const gsVideoUrl = plyUrl.replace('reconstructed_scene.ply', 'gs_trajectory.mp4');
                
                const baseDir = plyUrl.substring(0, plyUrl.lastIndexOf('/'));
                const parentDir = baseDir.substring(0, baseDir.lastIndexOf('/'));
                const stem = baseDir.substring(baseDir.lastIndexOf('/') + 1).replace('_gs_ours', '');
                const explorationVideoUrl = parentDir + '/' + stem + '.mp4';
                
                replayExplorationVideo.src = explorationVideoUrl;
                replayGsVideo.src = gsVideoUrl;
                replaysContainer.style.display = "block";
                
                // Trigger reload of video players
                replayExplorationVideo.load();
                replayGsVideo.load();
            } catch (e) {
                console.error("Failed to parse replay video URLs:", e);
                replaysContainer.style.display = "none";
            }
        }
    }
    
    // Create new viewer
    try {
        splatViewer = new GaussianSplats3D.Viewer({
            'sharedMemoryForWorkers': false,
            'selfDrivenMode': true,
            'rootElement': splatViewerContainer,
            'useBuiltInControls': true,
            'cameraUp': [0, 1, 0]
        });
        
        splatViewer.loadedUrl = plyUrl;
        
        splatViewer.addSplatScene(plyUrl, {
            'splatAlphaRemovalThreshold': 15,
            'showLoadingUI': false
        }).then(() => {
            if (splatLoading) splatLoading.classList.add("hidden");
            splatViewer.start();
        }).catch(err => {
            console.error("Failed to load splat scene:", err);
            if (splatLoading) {
                if (splatSpinner) splatSpinner.style.display = "none";
                if (splatLoadingText) splatLoadingText.innerText = "Error loading 3D environment.";
                if (splatLoadingSubtext) splatLoadingSubtext.innerText = "Please check the console or use the cinematic replays below.";
            }
        });
    } catch (err) {
        console.error("Failed to initialize GaussianSplats3D.Viewer:", err);
        if (splatLoading) {
            if (splatSpinner) splatSpinner.style.display = "none";
            if (splatLoadingText) splatLoadingText.innerText = "WebGL Initialization Failed.";
            if (splatLoadingSubtext) splatLoadingSubtext.innerText = "Your browser or system might not support WebGL. Please use the cinematic replays below to view the reconstructed 3D walkthroughs.";
        }
    }
}

// Load available trajectories list
async function loadTrajectories() {
    try {
        const res = await fetch("/api/trajectories");
        if (res.ok) {
            const list = await res.json();
            trajectorySelect.innerHTML = '<option value="preset" selected>Preset (Zoom-in / Zoom-out)</option>';
            list.forEach(name => {
                if (name !== "preset") {
                    const opt = document.createElement("option");
                    opt.value = name;
                    opt.innerText = `Custom: ${name}`;
                    trajectorySelect.appendChild(opt);
                }
            });
        }
    } catch (err) {
        console.error("Failed to load trajectories:", err);
    }
}

// Advanced toggle event
if (advancedToggle && advancedPanel) {
    advancedToggle.addEventListener("click", () => {
        const isOpen = advancedToggle.classList.toggle("open");
        advancedPanel.style.display = isOpen ? "grid" : "none";
    });
}

// Clear Terminal logs action
clearLogsBtn.addEventListener("click", () => {
    logTerminal.innerHTML = "";
});

// Stop pipeline action
if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
        appendTerminalLog("[SYSTEM] Requesting cancellation of the generation pipeline...");
        stopBtn.setAttribute("disabled", "true");
        try {
            const res = await fetch("/api/stop", { method: "POST" });
            if (res.ok) {
                appendTerminalLog("[SYSTEM] Cancel request sent to server.");
            } else {
                throw new Error(await res.text());
            }
        } catch (err) {
            appendTerminalLog(`[SYSTEM] ERROR: Failed to send cancel request: ${err.message}`);
            stopBtn.removeAttribute("disabled");
        }
    });
}

// Load PLY history function
async function loadPlyHistory() {
    try {
        const res = await fetch("/api/ply_files?t=" + Date.now());
        if (res.ok) {
            const list = await res.json();
            const currentVal = plyHistorySelect ? plyHistorySelect.value : "";
            if (plyHistorySelect) {
                plyHistorySelect.innerHTML = '<option value="">-- Select from History --</option>';
                list.forEach(item => {
                    const opt = document.createElement("option");
                    opt.value = item.url;
                    opt.innerText = item.label;
                    plyHistorySelect.appendChild(opt);
                });
                if (currentVal) {
                    plyHistorySelect.value = currentVal;
                } else if (list.length > 0) {
                    // Auto-load latest PLY on first load
                    const latestPly = list[0].url;
                    plyHistorySelect.value = latestPly;
                    initSplatViewer(latestPly);
                }
            }
        }
    } catch (err) {
        console.error("Failed to load PLY history:", err);
    }
}

// Local PLY file trigger
if (uploadPlyTrigger && localPlyInput) {
    uploadPlyTrigger.addEventListener("click", () => localPlyInput.click());
    
    localPlyInput.addEventListener("change", (e) => {
        if (localPlyInput.files.length) {
            const file = localPlyInput.files[0];
            if (!file.name.endsWith(".ply")) {
                alert("Please select a valid .ply Gaussian Splatting file.");
                return;
            }
            const objectUrl = URL.createObjectURL(file);
            appendTerminalLog(`[SYSTEM] Loading local PLY: ${file.name}...`);
            initSplatViewer(objectUrl);
            if (plyHistorySelect) plyHistorySelect.value = "";
        }
    });
}

// Select PLY from history
if (plyHistorySelect) {
    plyHistorySelect.addEventListener("change", () => {
        const selectedUrl = plyHistorySelect.value;
        if (selectedUrl) {
            appendTerminalLog(`[SYSTEM] Loading PLY from history: ${selectedUrl}...`);
            initSplatViewer(selectedUrl);
        }
    });
}

// Refresh PLY history
if (refreshPlyBtn) {
    refreshPlyBtn.addEventListener("click", () => {
        loadPlyHistory();
    });
}

// Stream to Rerun action
if (streamRerunBtn) {
    streamRerunBtn.addEventListener("click", async () => {
        const plyUrl = plyHistorySelect ? plyHistorySelect.value : "";
        if (!plyUrl) {
            alert("Please select a reconstructed scene to stream first.");
            return;
        }

        appendTerminalLog("[SYSTEM] Initiating stream to Rerun server...");
        streamRerunBtn.setAttribute("disabled", "true");
        streamRerunBtn.innerHTML = `
            <span class="spinner" style="width: 14px; height: 14px; border-width: 2px; margin-right: 6px;"></span>
            <span>Streaming...</span>
        `;

        const formData = new FormData();
        formData.append("ply_url", plyUrl);

        try {
            const res = await fetch("/api/stream_to_rerun", {
                method: "POST",
                body: formData
            });

            if (res.ok) {
                const data = await res.json();
                appendTerminalLog(`[SYSTEM] Rerun streaming background task started. Rerun Server IP: ${data.rerun_ip}`);
                connectProgressStream();
            } else {
                throw new Error(await res.text());
            }
        } catch (err) {
            appendTerminalLog(`[SYSTEM] ERROR: Failed to stream to Rerun: ${err.message}`);
        } finally {
            // Restore button after a short delay
            setTimeout(() => {
                streamRerunBtn.removeAttribute("disabled");
                streamRerunBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 16px; height: 16px;">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                    <span>Stream to Rerun</span>
                `;
            }, 3000);
        }
    });
}

// Initialize on page load
connectMetricsStream();
loadTrajectories();
loadPlyHistory();

