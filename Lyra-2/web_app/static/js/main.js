// State Variables
let uploadedFilename = "";
let progressEventSource = null;
let metricsEventSource = null;

// DOM Elements
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const imagePreview = document.getElementById("image-preview");
const dropZoneContent = document.querySelector(".drop-zone-content");
const generateBtn = document.getElementById("generate-btn");
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

// Resource Card Value Labels
const gpuUtilVal = document.getElementById("gpu-util-val");
const gpuMemVal = document.getElementById("gpu-mem-val");
const cpuUtilVal = document.getElementById("cpu-util-val");
const ramUtilVal = document.getElementById("ram-util-val");

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
        cpuUtilVal.innerText = `${data.cpu_util}%`;
        ramUtilVal.innerText = `${data.ram_util}%`;
        
        // Update Charts
        updateChartData(charts.gpu, data.gpu_util);
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
    
    // Disable inputs
    generateBtn.setAttribute("disabled", "true");
    promptInput.setAttribute("disabled", "true");
    fileInput.setAttribute("disabled", "true");
    
    // Clear logs and hide old outputs
    logTerminal.innerHTML = "";
    outputCard.style.display = "none";
    
    appendTerminalLog("[SYSTEM] Initializing 3D Generative Pipeline...\n");
    
    // Reset steps UI
    updateStepStatus(stepVideo, "pending");
    updateStepStatus(stepRecon, "pending");
    
    // Trigger run
    const formData = new FormData();
    formData.append("filename", uploadedFilename);
    formData.append("prompt", prompt);
    
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
        generateBtn.removeAttribute("disabled");
        promptInput.removeAttribute("disabled");
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
                enableInputs();
            } else if (status.stage === "failed") {
                // Find active step and mark failed
                if (stepVideo.classList.contains("active")) {
                    updateStepStatus(stepVideo, "failed");
                } else if (stepRecon.classList.contains("active")) {
                    updateStepStatus(stepRecon, "failed");
                }
                appendTerminalLog(`\n[SYSTEM] ERROR: Pipeline failed: ${status.error}\n`);
                progressEventSource.close();
                enableInputs();
            }
        }
    };
    
    progressEventSource.onerror = () => {
        console.error("Progress EventSource disconnected.");
    };
}

// Helper: Enable Inputs after run
function enableInputs() {
    generateBtn.removeAttribute("disabled");
    promptInput.removeAttribute("disabled");
    fileInput.removeAttribute("disabled");
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
    downloadPlyBtn.href = plyUrl;
    outputCard.style.display = "block";
    
    // Scroll output card into view
    outputCard.scrollIntoView({ behavior: "smooth" });
}

// Clear Terminal logs action
clearLogsBtn.addEventListener("click", () => {
    logTerminal.innerHTML = "";
});

// Initialize on page load
connectMetricsStream();
