#!/usr/bin/env bash
#
# Control script for Project Lyra 2.0 Web Dashboard & Docker Container
# Script de contrôle pour le tableau de bord Lyra 2.0 et son conteneur Docker
#

set -euo pipefail

# Determine repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="lyra2_dev"
PORT=7860
SYMLINK_PATH="/home/sparka/Generative 3D environment from picture"

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper functions for logs
log_info() {
    echo -e "${BLUE}${BOLD}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}${BOLD}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}${BOLD}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}${BOLD}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
}

# Check GPU status
check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        log_success "NVIDIA GPU Detected: ${CYAN}${GPU_NAME}${NC}"
    else
        log_warn "No NVIDIA GPU detected on host. Running without GPU support may fail."
    fi
}

# Check checkpoints status
check_checkpoints() {
    echo -e "\n${BOLD}--- Checkpoints Status / État des modèles ---${NC}"
    local ckpt_dir="${REPO_ROOT}/Lyra-2/checkpoints"
    local required_subdirs=("model" "vae" "text_encoder" "image_encoder" "recon")
    local missing=0

    if [ ! -d "$ckpt_dir" ]; then
        log_warn "Checkpoints directory not found at: ${ckpt_dir}"
        missing=1
    else
        for dir in "${required_subdirs[@]}"; do
            if [ -d "${ckpt_dir}/${dir}" ] && [ "$(ls -A "${ckpt_dir}/${dir}")" ]; then
                echo -e "  [✓] ${dir} - ${GREEN}Downloaded / Téléchargé${NC}"
            else
                echo -e "  [✗] ${dir} - ${RED}Missing / Manquant${NC}"
                missing=$((missing + 1))
            fi
        done
    fi

    if [ $missing -gt 0 ]; then
        log_warn "Some models are missing. You can download them with:"
        echo -e "  huggingface-cli download nvidia/Lyra-2.0 --include \"checkpoints/*\" --local-dir ${REPO_ROOT}/Lyra-2"
    else
        log_success "All main model checkpoints are present."
    fi
}

# Setup container dependencies
setup_dependencies() {
    log_info "Compiling and installing Lyra 2.0 dependencies inside the container..."
    log_info "This installs CUDA extensions (flash-attn, vipe, depth_anything_3[gs])..."
    log_info "This may take several minutes. Running compilation script..."
    
    docker exec -it "${CONTAINER_NAME}" bash -c "/workspace/lyra/compile_install.sh"
    
    log_success "Setup complete! All dependencies compiled and installed."
}

# Start everything
start_all() {
    check_docker
    check_gpu

    echo -e "\n${BOLD}--- Starting Services / Démarrage des Services ---${NC}"

    # 1. Manage Symlink (Ensuring container's bind mount has correct source folder)
    if [ ! -L "${SYMLINK_PATH}" ] || [ "$(readlink -f "${SYMLINK_PATH}")" != "${REPO_ROOT}" ]; then
        log_info "Ensuring compatibility path symlink at: ${SYMLINK_PATH}"
        # Remove if it exists but is wrong/empty directory
        if [ -d "${SYMLINK_PATH}" ] && [ ! -L "${SYMLINK_PATH}" ]; then
            sudo rmdir "${SYMLINK_PATH}" || sudo rm -rf "${SYMLINK_PATH}"
        elif [ -L "${SYMLINK_PATH}" ]; then
            rm -f "${SYMLINK_PATH}"
        fi
        ln -sf "${REPO_ROOT}" "${SYMLINK_PATH}"
    fi

    # 2. Check and start Docker container
    local container_exists
    container_exists=$(docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}')

    if [ -z "${container_exists}" ]; then
        log_info "Container '${CONTAINER_NAME}' does not exist. Creating new container..."
        docker run --gpus all -d -it --name "${CONTAINER_NAME}" \
          -v "${SYMLINK_PATH}:/workspace/lyra" \
          -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
          -p "${PORT}:${PORT}" \
          nvcr.io/nvidia/pytorch:25.01-py3
        log_success "Container created."
        setup_dependencies
    else
        # Check if container is running
        local is_running
        is_running=$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}')
        if [ -z "${is_running}" ]; then
            log_info "Starting stopped container '${CONTAINER_NAME}'..."
            docker start "${CONTAINER_NAME}"
            log_success "Container started."
        else
            log_success "Container '${CONTAINER_NAME}' is already running."
        fi
    fi

    # 3. Check if dependencies need compilation (fallback)
    if ! docker exec "${CONTAINER_NAME}" python3 -c "import vipe_ext" &>/dev/null; then
        log_warn "Extensions not found in the container. Automatically running setup..."
        setup_dependencies
    fi

    # 4. Start the FastAPI Web Dashboard
    local is_dashboard_running
    is_dashboard_running=$(docker exec "${CONTAINER_NAME}" ps aux | grep -v grep | grep "web_app/app.py" || true)

    if [ -n "${is_dashboard_running}" ]; then
        log_success "Web Dashboard is already running."
    else
        log_info "Starting Web Dashboard inside container..."
        docker exec -d "${CONTAINER_NAME}" bash -c "export LD_PRELOAD=/workspace/lyra/libcuda_fake.so && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export PYTHONPATH=/workspace/lyra/Lyra-2 && python3 /workspace/lyra/Lyra-2/web_app/app.py > /workspace/lyra/web_app.log 2>&1"
        
        # Wait a moment for dashboard to start
        sleep 2
        
        # Verify it started
        if docker exec "${CONTAINER_NAME}" ps aux | grep -v grep | grep "web_app/app.py" &>/dev/null; then
            log_success "Web Dashboard started successfully."
        else
            log_error "Web Dashboard failed to start. Check logs with: ./manage.sh logs"
            exit 1
        fi
    fi

    echo -e "\n${GREEN}${BOLD}====================================================${NC}"
    echo -e "${GREEN}${BOLD} Lyra 2.0 is up and running! / Lyra 2.0 est démarré!${NC}"
    echo -e "${BOLD} Access URL:${NC} http://localhost:${PORT}"
    echo -e " Check logs:   ./manage.sh logs"
    echo -e " Stop services: ./manage.sh stop"
    echo -e "${GREEN}${BOLD}====================================================${NC}\n"
}

# Stop everything
stop_all() {
    check_docker
    echo -e "\n${BOLD}--- Stopping Services / Arrêt des Services ---${NC}"

    local container_exists
    container_exists=$(docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}')

    if [ -z "${container_exists}" ]; then
        log_warn "Container '${CONTAINER_NAME}' does not exist. Nothing to stop."
        return
    fi

    local is_running
    is_running=$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}')

    if [ -n "${is_running}" ]; then
        # Kill the dashboard process
        log_info "Stopping Web Dashboard process inside container..."
        docker exec "${CONTAINER_NAME}" pkill -f "web_app/app.py" || true
        log_success "Web Dashboard stopped."

        # Stop container
        log_info "Stopping Docker container '${CONTAINER_NAME}'..."
        docker stop "${CONTAINER_NAME}"
        log_success "Container stopped."
    else
        log_success "Container is already stopped."
    fi
}

# Show status
show_status() {
    check_docker
    echo -e "\n${BOLD}--- System Status / État du Système ---${NC}"
    
    # 1. GPU Info
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader | while read -r line; do
            echo -e "  GPU: ${CYAN}${line}${NC}"
        done
    else
        echo -e "  GPU: ${RED}No NVIDIA GPU found on host${NC}"
    fi

    # 2. Container Status
    local container_status
    container_status=$(docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Status}}')
    if [ -z "${container_status}" ]; then
        echo -e "  Docker Container: ${RED}Not Created / Non créé${NC}"
    elif [[ "${container_status}" == Up* ]]; then
        echo -e "  Docker Container: ${GREEN}Running / En cours d'exécution (${container_status})${NC}"
    else
        echo -e "  Docker Container: ${YELLOW}Stopped / Arrêté (${container_status})${NC}"
    fi

    # 3. Web Dashboard Status
    if [ -n "${container_status}" ] && [[ "${container_status}" == Up* ]]; then
        local is_dashboard_running
        is_dashboard_running=$(docker exec "${CONTAINER_NAME}" ps aux | grep -v grep | grep "web_app/app.py" || true)
        if [ -n "${is_dashboard_running}" ]; then
            echo -e "  Web Dashboard: ${GREEN}Active / En cours (Port ${PORT})${NC}"
        else
            echo -e "  Web Dashboard: ${RED}Inactive / Non démarré${NC}"
        fi
    else
        echo -e "  Web Dashboard: ${RED}Inactive (Container Stopped)${NC}"
    fi

    # 4. Checkpoints
    check_checkpoints
    echo ""
}

# Show logs
show_logs() {
    check_docker
    local is_running
    is_running=$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}')
    
    if [ -z "${is_running}" ]; then
        log_error "Container '${CONTAINER_NAME}' is not running. Cannot view logs."
        exit 1
    fi

    log_info "Tailing Web Dashboard logs (Press Ctrl+C to exit)..."
    docker exec -it "${CONTAINER_NAME}" tail -f /workspace/lyra/web_app.log || true
}

# Main routing
case "${1:-}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        start_all
        ;;
    status)
        show_status
        ;;
    setup)
        check_docker
        setup_dependencies
        ;;
    logs)
        show_logs
        ;;
    *)
        echo -e "${BOLD}Usage:${NC} $0 {start|stop|restart|status|setup|logs}"
        echo -e "\n${BOLD}Commands:${NC}"
        echo -e "  ${CYAN}start${NC}   : Start Docker container & Web Dashboard"
        echo -e "  ${CYAN}stop${NC}    : Stop Web Dashboard & Docker container"
        echo -e "  ${CYAN}restart${NC} : Restart all services"
        echo -e "  ${CYAN}status${NC}  : Check running services, GPU, and model files"
        echo -e "  ${CYAN}setup${NC}   : Re-run compilation/dependencies setup inside container"
        echo -e "  ${CYAN}logs${NC}    : View live Web Dashboard server logs"
        exit 1
        ;;
esac
