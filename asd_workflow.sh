#!/bin/bash

# Сначала чиним путь к ComfyUI (КРИТИЧНО для Vast.ai)
if [ ! -f "/workspace/ComfyUI/main.py" ]; then
    echo "[Fix] ComfyUI missing in workspace. Linking..."
    rm -rf /workspace/ComfyUI
    ln -s /opt/workspace-internal/ComfyUI /workspace/ComfyUI
fi

BASE="/workspace/ComfyUI/models"

# Функция для скачивания
get() {
    local url="$1"
    local folder="$2"
    local file=$(basename "$url")

    echo ">>> Downloading: $file"
    wget -nc --show-progress "$url" -O "$folder/$file"
}

echo "===================================================="
echo " DOWNLOADING WAN 2.2 PACK + PATCHES"
echo "===================================================="


# ---------------------------------------------------------
# Diffusion models
# ---------------------------------------------------------
echo ">>> Diffusion Models..."
get "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors" \
    "$BASE/diffusion_models"


# ---------------------------------------------------------
# LoRAs
# ---------------------------------------------------------
echo ">>> LoRAs..."

get "https://huggingface.co/rahul7star/wan2.2Lora/resolve/main/BounceHighWan2_2.safetensors" \
    "$BASE/loras"

get "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors" \
    "$BASE/loras"


# ---------------------------------------------------------
# VAE
# ---------------------------------------------------------
echo ">>> VAE..."
get "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    "$BASE/vae"


# ---------------------------------------------------------
# CLIP Vision
# ---------------------------------------------------------
echo ">>> CLIP Vision..."
get "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/model.safetensors" \
    "$BASE/clip_vision"


# ---------------------------------------------------------
# Upscale model
# ---------------------------------------------------------
echo ">>> Upscale..."
get "https://huggingface.co/dtarnow/UPscaler/resolve/main/RealESRGAN_x2plus.pth" \
    "$BASE/upscale_models"

get "https://huggingface.co/WedManHK/test2/resolve/20c1bfd934423c265890d1084d548837a68b56ae/2xNomosUni_span_multijpg.pth" \
    "$BASE/upscale_models"
# ---------------------------------------------------------
# Detection models (create folder!)
# ---------------------------------------------------------
echo ">>> Creating detection folder..."
mkdir -p "$BASE/detection"

echo ">>> Detection models..."
get "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx" \
    "$BASE/detection"

get "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx" \
    "$BASE/detection"

# ---------------------------------------------------------
# Detection models (create folder!)
# ---------------------------------------------------------

get "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$BASE/text_encoders"

# # # ---------------------------------------------------------
# # # SageAttention reinstall
# # # ---------------------------------------------------------
echo ">>> Reinstalling SageAttention..."
pip install flash-attn --no-build-isolation


# ---------------------------------------------------------
# Patch config.ini (security_level = weak)
# ---------------------------------------------------------
CFG="/workspace/ComfyUI/user/__manager/config.ini"

if [ -f "$CFG" ]; then
    echo ">>> Patching config.ini (security_level = weak)..."
    sed -i 's/security_level *= *normal/security_level = weak/g' "$CFG"
else
    echo "WARNING: config.ini not found: $CFG"
fi

# Настройка путей
COMFY_NODES_DIR="/workspace/ComfyUI/custom_nodes"
mkdir -p "$COMFY_NODES_DIR"
cd "$COMFY_NODES_DIR"

echo "==============================================="
echo "STARTING CUSTOM NODES INSTALLATION TEST"
echo "==============================================="

# Список репозиториев
REPOS=(
    "https://github.com/kijai/ComfyUI-WanAnimatePreprocess"
    "https://github.com/storyicon/comfyui_segment_anything"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/Azornes/Comfyui-Resolution-Master"
)

# 1. Клонирование репозиториев
for repo in "${REPOS[@]}"; do
    folder=$(basename "$repo" .git)
    if [ ! -d "$folder" ]; then
        echo ">>> Cloning $folder..."
        git clone "$repo"
    else
        echo ">>> $folder already exists, pulling updates..."
        cd "$folder" && git pull && cd ..
    fi
done

echo "-----------------------------------------------"
echo "INSTALLING DEPENDENCIES"
echo "-----------------------------------------------"

# Обновляем pip перед установкой
pip install --upgrade pip setuptools wheel

# Внутри download_models.sh вместо обычного pip install:
MAIN_PIP="/venv/main/bin/pip"
for folder in /workspace/ComfyUI/custom_nodes/*; do
    if [ -f "$folder/requirements.txt" ]; then
        $MAIN_PIP install --no-cache-dir -r "$folder/requirements.txt" --no-build-isolation
        if [ $? -eq 0 ]; then
            echo " [OK] $folder dependencies installed."
        else
            echo " [ERROR] Failed to install dependencies for $folder"
        fi
    else
        echo ">>> No requirements.txt in $folder, skipping."
    fi
done

echo "==============================================="
echo " RESTARTING COMFYUI"
echo "==============================================="

# 1. Находим PID процесса ComfyUI (ищем main.py)
PID=$(pgrep -f "python3 main.py" || pgrep -f "python main.py")

if [ ! -z "$PID" ]; then
    echo ">>> Found ComfyUI process (PID: $PID). Terminating..."
    kill -9 $PID
    sleep 3
else
    echo ">>> ComfyUI process not found. Starting fresh..."
fi

# Скрипт для запуска ComfyUI на всех GPU Vast.ai
# Использование: ./launch_comfy_multi.sh [ports_file]
# ports_file - файл со списком портов (по одному на строку), по умолчанию ports.txt

PORTS_FILE="${1:-ports.txt}"
BASE_PORT=2818
PORTS_PER_GPU=1  # Если больше, генерируем список

# Создаём список портов, если файла нет
if [ ! -f "$PORTS_FILE" ]; then
    echo "Создаём ports.txt с портами: 2818 3818 4818 5818 6818"
    cat > "$PORTS_FILE" << EOF
2818
3818
4818
5818
6818
7818
8818
9818
EOF
fi

# Получаем количество GPU
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
echo "Найдено GPU: $NUM_GPUS"

# Читаем порты
mapfile -t PORTS < "$PORTS_FILE"
NUM_PORTS=${#PORTS[@]}
echo "Доступно портов: $NUM_PORTS"

if [ $NUM_GPUS -gt $NUM_PORTS ]; then
    echo "Предупреждение: GPU ($NUM_GPUS) > портов ($NUM_PORTS). Используем первые $NUM_PORTS GPU."
    NUM_INSTANCES=$NUM_PORTS
else
    NUM_INSTANCES=$NUM_GPUS
fi

echo "Запускаем $NUM_INSTANCES инстансов ComfyUI..."

# Массив PID для процессов
PIDS=()

# Исправленный блок запуска внутри asd_workflow.sh
for ((i=0; i<$NUM_INSTANCES; i++)); do
    GPU_ID=$i
    PORT=${PORTS[$i]}
    
    echo "Запуск GPU$GPU_ID на порту $PORT..."
    
    # ПРОВЕРКА: существует ли файл вообще
    if [ ! -f "/workspace/ComfyUI/main.py" ]; then
        echo "КРИТИЧЕСКАЯ ОШИБКА: /workspace/ComfyUI/main.py не найден!"
        # Пытаемся восстановить ссылку, если её нет
        ln -s /opt/workspace-internal/ComfyUI /workspace/ComfyUI
    fi

    # Команда запуска с ПРАВИЛЬНЫМ путем к Python и абсолютными путями
    # Используем /venv/main/bin/python, чтобы подхватились все зависимости
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID /venv/main/bin/python /workspace/ComfyUI/main.py \
        --port $PORT \
        --listen 0.0.0.0 \
        --disable-auto-launch \
        --enable-cors-header"
    
    # Запуск с перенаправлением ошибок (2>&1) крайне важен
    nohup bash -c "$CMD" > "/workspace/comfyui_gpu${GPU_ID}_port${PORT}.log" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    sleep 2 # Даем немного времени на инициализацию перед следующим GPU
done

echo ""
echo "Все инстансы запущены! ПИДы: ${PIDS[*]}"
echo "Логи: comfyui_gpu*_port*.log"
echo ""
echo "Для остановки всех:"
echo "kill \${PIDS[*]}  # или вручную kill PID"
echo "Для статуса: nvidia-smi && ps aux | grep main.py"

# Сохраняем PIDS в файл
printf '%s\n' "${PIDS[@]}" > comfyui_pids.txt
echo "PIDS сохранены в comfyui_pids.txt"

echo ""
echo "==============================================="
echo " SETTING UP TELEGRAM BOT"
echo "==============================================="

BOT_DIR="/workspace/characterSwap"
mkdir -p "$BOT_DIR"
cd "$BOT_DIR"

# Скачиваем .env файл
echo ">>> Downloading .env file..."
wget -O "$BOT_DIR/.env" "https://www.dropbox.com/scl/fi/40cjh3eli3hwp51s0c682/.env?rlkey=abo3by43ahkhdch3gidvs992u&st=aaekg7cm&dl=1"

# Создаем venv для бота
if [ ! -d "$BOT_DIR/venv" ]; then
    echo ">>> Creating virtual environment for bot..."
    python3 -m venv venv
else
    echo ">>> Virtual environment already exists"
fi

# Активируем venv
source "$BOT_DIR/venv/bin/activate"

# Устанавливаем зависимости если есть requirements.txt
if [ -f "$BOT_DIR/requirements.txt" ]; then
    echo ">>> Installing bot dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo ">>> No requirements.txt found, installing default dependencies..."
    pip install aiogram aiohttp python-dotenv
fi

# Запускаем бота в фоне
if [ -f "$BOT_DIR/bot.py" ]; then
    echo ">>> Starting Telegram bot..."
    nohup python bot.py > "$BOT_DIR/bot.log" 2>&1 &
    BOT_PID=$!
    echo "Bot PID: $BOT_PID"
    echo $BOT_PID > "$BOT_DIR/bot_pid.txt"
    echo "Bot started! Log: $BOT_DIR/bot.log"
else
    echo "ERROR: bot.py not found in $BOT_DIR"
fi

# Деактивируем venv
deactivate

echo ""
echo "==============================================="
echo " ALL SERVICES STARTED"
echo "==============================================="
echo "ComfyUI instances: ${PIDS[*]}"
echo "Bot PID: $BOT_PID"
echo ""
echo "To stop bot: kill $BOT_PID"
echo "To view bot log: tail -f $BOT_DIR/bot.log"