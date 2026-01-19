"""
Telegram Bot for ComfyUI - Motion Transfer
Photo + Video = Animated Photo
Multi-GPU Support with Port Rotation
FIXED VERSION: Added error handling for face detection and video duration issues
"""
import os
import asyncio
import aiohttp
import json
import uuid
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    ReplyKeyboardMarkup, 
    KeyboardButton, 
    ReplyKeyboardRemove, 
    FSInputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
COMFY_BASE_HOST = os.getenv("COMFY_BASE_HOST", "http://127.0.0.1")
INPUT_DIR = os.getenv("INPUT_DIR", "/workspace/ComfyUI/input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/ComfyUI/output")

# Список портов для проверки
COMFY_PORTS = [2818, 3818, 4818, 5818, 6818, 7818, 8818, 9818]

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

WHITELIST_FILE = "whitelist.txt"

RESOLUTIONS = {
    "512x512": (512, 512),
    "640x640": (640, 640),
    "720x1280": (720, 1280),
    "1280x720": (1280, 720),
    "432x864": (432, 864),
    "864x432": (864, 432)
}


def load_whitelist():
    if not os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, "w") as f:
            f.write(f"{ADMIN_ID}\n")
        return {ADMIN_ID}
    with open(WHITELIST_FILE, "r") as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def save_whitelist(whitelist):
    with open(WHITELIST_FILE, "w") as f:
        for user_id in sorted(whitelist):
            f.write(f"{user_id}\n")


whitelist = load_whitelist()


def check_access(user_id):
    return user_id in whitelist


def is_admin(user_id):
    return user_id == ADMIN_ID


@dataclass
class QueueTask:
    task_id: str
    user_id: int
    username: str
    photo_filename: str
    video_filename: str
    width: int
    height: int
    port: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "waiting"
    prompt_id: Optional[str] = None
    result_file: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0  # NEW: Track retry attempts


class PortManager:
    """Менеджер для работы с портами ComfyUI"""
    
    def __init__(self, base_host: str, ports: List[int]):
        self.base_host = base_host
        self.all_ports = ports
        self.active_ports = []
        self.current_port_index = 0
        self._lock = asyncio.Lock()
    
    async def detect_active_ports(self):
        """Автоматически определяет активные порты ComfyUI"""
        active = []
        
        print("Detecting active ComfyUI ports...")
        
        for port in self.all_ports:
            url = f"{self.base_host}:{port}"
            if await self.check_port_available(url):
                active.append(port)
                print(f"✓ Port {port} is active")
            else:
                print(f"✗ Port {port} is not available")
        
        async with self._lock:
            self.active_ports = active
            self.current_port_index = 0
        
        print(f"\nTotal active ports: {len(active)}")
        return active
    
    async def check_port_available(self, url: str, timeout: float = 3.0) -> bool:
        """Проверяет доступность порта ComfyUI"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def get_next_port(self) -> Optional[int]:
        """Возвращает следующий порт по кругу (ротация)"""
        async with self._lock:
            if not self.active_ports:
                return None
            
            port = self.active_ports[self.current_port_index]
            self.current_port_index = (self.current_port_index + 1) % len(self.active_ports)
            return port
    
    def get_url_for_port(self, port: int) -> str:
        """Возвращает полный URL для порта"""
        return f"{self.base_host}:{port}"
    
    async def get_active_ports(self) -> List[int]:
        """Возвращает список активных портов"""
        async with self._lock:
            return self.active_ports.copy()


class PortQueueManager:
    """Менеджер очередей для каждого порта"""
    
    def __init__(self, port: int, comfy_url: str):
        self.port = port
        self.comfy_url = comfy_url
        self.queue = deque()
        self.current_task = None
        self.is_processing = False
        self._lock = asyncio.Lock()
    
    async def add_task(self, task: QueueTask):
        """Добавляет задачу в очередь порта"""
        async with self._lock:
            task.port = self.port
            self.queue.append(task)
            return len(self.queue)
    
    async def get_next_task(self) -> Optional[QueueTask]:
        """Получает следующую задачу из очереди"""
        async with self._lock:
            if self.queue:
                return self.queue.popleft()
            return None
    
    async def set_current(self, task: Optional[QueueTask]):
        """Устанавливает текущую обрабатываемую задачу"""
        async with self._lock:
            self.current_task = task
            self.is_processing = task is not None
    
    async def get_queue_length(self) -> int:
        """Возвращает длину очереди"""
        async with self._lock:
            return len(self.queue)
    
    async def is_busy(self) -> bool:
        """Проверяет, занят ли порт обработкой"""
        async with self._lock:
            return self.is_processing
    
    async def cancel_task(self, task_id: str, user_id: int):
        """Отменяет задачу из очереди"""
        async with self._lock:
            if self.current_task and self.current_task.task_id == task_id:
                if self.current_task.user_id != user_id and user_id != ADMIN_ID:
                    return False, "You can only cancel your own tasks"
                return True, "processing"
            
            for task in list(self.queue):
                if task.task_id == task_id:
                    if task.user_id != user_id and user_id != ADMIN_ID:
                        return False, "You can only cancel your own tasks"
                    self.queue.remove(task)
                    cleanup_task_files(task)
                    return True, "removed"
            
            return False, "Task not found"


class GlobalQueueManager:
    """Глобальный менеджер очередей для всех портов"""
    
    def __init__(self, port_manager: PortManager):
        self.port_manager = port_manager
        self.port_queues = {}  # {port: PortQueueManager}
        self._lock = asyncio.Lock()
    
    async def initialize_queues(self):
        """Инициализирует очереди для всех активных портов"""
        active_ports = await self.port_manager.get_active_ports()
        
        async with self._lock:
            for port in active_ports:
                if port not in self.port_queues:
                    comfy_url = self.port_manager.get_url_for_port(port)
                    self.port_queues[port] = PortQueueManager(port, comfy_url)
        
        print(f"Initialized queues for {len(self.port_queues)} ports")
    
    async def add_task(self, task: QueueTask) -> tuple[int, int]:
        """
        Добавляет задачу в очередь следующего доступного порта
        Возвращает: (port, position_in_queue)
        """
        # Получаем следующий порт по ротации
        port = await self.port_manager.get_next_port()
        
        if port is None:
            raise Exception("No active ComfyUI ports available")
        
        # Добавляем задачу в очередь этого порта
        port_queue = self.port_queues.get(port)
        if port_queue is None:
            raise Exception(f"Queue for port {port} not initialized")
        
        position = await port_queue.add_task(task)
        return port, position
    
    async def get_all_user_tasks(self, user_id: int) -> List[QueueTask]:
        """Получает все задачи пользователя из всех очередей"""
        all_tasks = []
        
        for port, port_queue in self.port_queues.items():
            async with port_queue._lock:
                # Добавляем текущую задачу
                if port_queue.current_task and port_queue.current_task.user_id == user_id:
                    all_tasks.append(port_queue.current_task)
                
                # Добавляем задачи из очереди
                for task in port_queue.queue:
                    if task.user_id == user_id:
                        all_tasks.append(task)
        
        # Сортируем по времени создания
        all_tasks.sort(key=lambda t: t.created_at)
        return all_tasks
    
    async def cancel_task(self, task_id: str, user_id: int):
        """Отменяет задачу из любой очереди"""
        for port, port_queue in self.port_queues.items():
            success, status = await port_queue.cancel_task(task_id, user_id)
            if success or status != "Task not found":
                return success, status
        
        return False, "Task not found"
    
    async def get_global_status(self) -> dict:
        """Получает общий статус всех очередей"""
        total_queue = 0
        processing_count = 0
        port_status = {}
        
        for port, port_queue in self.port_queues.items():
            queue_len = await port_queue.get_queue_length()
            is_busy = await port_queue.is_busy()
            
            total_queue += queue_len
            if is_busy:
                processing_count += 1
            
            port_status[port] = {
                "queue_length": queue_len,
                "is_processing": is_busy
            }
        
        return {
            "total_ports": len(self.port_queues),
            "processing_ports": processing_count,
            "total_queue": total_queue,
            "port_status": port_status
        }


def cleanup_task_files(task):
    try:
        photo_path = os.path.join(INPUT_DIR, task.photo_filename)
        video_path = os.path.join(INPUT_DIR, task.video_filename)
        if os.path.exists(photo_path):
            os.remove(photo_path)
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception as e:
        print(f"Error cleaning files: {e}")


class GenerationStates(StatesGroup):
    choosing_resolution = State()
    waiting_photo = State()
    waiting_video = State()


bot = Bot(token=TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# Инициализируем менеджеры портов и очередей
port_manager = PortManager(COMFY_BASE_HOST, COMFY_PORTS)
queue_manager = GlobalQueueManager(port_manager)


def get_resolution_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="512x512"), KeyboardButton(text="640x640")],
            [KeyboardButton(text="720x1280"), KeyboardButton(text="1280x720")],
            [KeyboardButton(text="432x864"), KeyboardButton(text="864x432")]
        ],
        resize_keyboard=True
    )


def get_cancel_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Cancel")]],
        resize_keyboard=True
    )


def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="New Generation")],
            [KeyboardButton(text="My Tasks"), KeyboardButton(text="Queue Status")]
        ],
        resize_keyboard=True
    )


def load_workflow():
    with open("workflow.json", "r", encoding="utf-8") as f:
        return json.load(f)


def modify_workflow(workflow, task):
    """
    FIXED: Модифицирует workflow с исправлениями для длительности видео
    """
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        title = node.get("_meta", {}).get("title", "")
        
        # LoadImage - photo (face)
        if class_type == "LoadImage":
            node["inputs"]["image"] = task.photo_filename
        
        # VHS_LoadVideo - video (motion source)
        elif class_type == "VHS_LoadVideo":
            node["inputs"]["video"] = task.video_filename
        
        # Resolution Master (ID 884)
        elif class_type == "ResolutionMaster":
            node["inputs"]["width"] = task.width
            node["inputs"]["height"] = task.height
        
        # КРИТИЧЕСКИ ВАЖНО: Отключаем trim_to_audio для VHS_VideoCombine
        # Это предотвращает удлинение видео из-за аудио
        elif class_type == "VHS_VideoCombine":
            if "trim_to_audio" in node["inputs"]:
                node["inputs"]["trim_to_audio"] = False
                print(f"[WORKFLOW FIX] Disabled trim_to_audio for node {node_id}")
    
    return workflow


async def process_port_queue(port: int, port_queue: PortQueueManager):
    """Обработчик очереди для конкретного порта"""
    print(f"Started queue processor for port {port}")
    
    while True:
        try:
            is_busy = await port_queue.is_busy()
            queue_len = await port_queue.get_queue_length()
            
            if not is_busy and queue_len > 0:
                task = await port_queue.get_next_task()
                if task:
                    await process_task(task, port_queue)
            
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Error in process_port_queue (port {port}): {e}")
            await asyncio.sleep(5)


async def interrupt_comfy_processing(comfy_url: str):
    """Прерывает текущую обработку на ComfyUI сервере"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{comfy_url}/interrupt", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
    except Exception as e:
        print(f"Error interrupting ComfyUI: {e}")
        return False


async def process_task(task: QueueTask, port_queue: PortQueueManager):
    """
    FIXED: Обработка задачи с улучшенной обработкой ошибок
    """
    await port_queue.set_current(task)
    task.status = "processing"
    
    comfy_url = port_queue.comfy_url
    MAX_RETRIES = 2  # Максимум 2 повторные попытки
    
    try:
        await bot.send_message(
            task.user_id,
            f"🎬 Processing started!\n"
            f"Port: {task.port}\n"
            f"Resolution: {task.width}x{task.height}"
        )
        
        workflow = load_workflow()
        workflow = modify_workflow(workflow, task)
        
        async with aiohttp.ClientSession() as session:
            # Отправляем workflow
            payload = {"prompt": workflow, "client_id": f"bot_{task.task_id}"}
            async with session.post(f"{comfy_url}/prompt", json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to queue prompt: {resp.status}")
                
                result = await resp.json()
                prompt_id = result.get("prompt_id")
                task.prompt_id = prompt_id
                
                if not prompt_id:
                    raise Exception("No prompt_id in response")
            
            # Ожидаем завершения с обработкой ошибок
            completion_status = await wait_for_completion(session, comfy_url, prompt_id, task.user_id)
            
            # Проверяем на критические ошибки NaN
            if completion_status and "error" in completion_status:
                error_msg = completion_status["error"]
                
                # Проверяем на ошибку "cannot convert float NaN to integer"
                if "NaN" in error_msg or "cannot convert float" in error_msg:
                    # Это ошибка детекции лица - лицо не обнаружено четко
                    if task.retry_count < MAX_RETRIES:
                        task.retry_count += 1
                        print(f"[RETRY] Face detection NaN error, retry {task.retry_count}/{MAX_RETRIES}")
                        
                        # Прерываем текущую обработку
                        await interrupt_comfy_processing(comfy_url)
                        await asyncio.sleep(3)
                        
                        # Добавляем задачу обратно в очередь
                        await port_queue.add_task(task)
                        await port_queue.set_current(None)
                        
                        await bot.send_message(
                            task.user_id,
                            f"⚠️ Face detection issue (attempt {task.retry_count}/{MAX_RETRIES})\n"
                            f"Retrying... Please ensure the face is clearly visible in the photo."
                        )
                        return
                    else:
                        # Исчерпаны попытки
                        raise Exception(
                            "Face detection failed after multiple attempts. "
                            "Please ensure:\n"
                            "1. The face is clearly visible\n"
                            "2. Good lighting on the face\n"
                            "3. Face is not too small or too large\n"
                            "4. Face is looking towards camera"
                        )
                
                # Другие ошибки - пробрасываем дальше
                raise Exception(error_msg)
            
            # Получаем результат
            output_file = await get_output_file(prompt_id)
            
            if output_file and os.path.exists(output_file):
                task.result_file = output_file
                task.status = "completed"
                
                video = FSInputFile(output_file)
                await bot.send_video(
                    task.user_id,
                    video,
                    caption=f"✅ Generation complete!\nPort: {task.port}"
                )
                
                # Очистка файлов
                cleanup_task_files(task)
                try:
                    os.remove(output_file)
                except Exception as e:
                    print(f"Error removing output: {e}")
            else:
                raise Exception("Output file not found")
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing task {task.task_id}: {error_msg}")
        task.status = "error"
        task.error = error_msg
        
        # Прерываем обработку на сервере
        await interrupt_comfy_processing(comfy_url)
        
        try:
            await bot.send_message(
                task.user_id,
                f"❌ Error during generation:\n{error_msg[:500]}"
            )
        except Exception:
            pass
        
        cleanup_task_files(task)
    
    finally:
        await port_queue.set_current(None)


async def wait_for_completion(session, comfy_url, prompt_id, user_id, timeout=12000):
    """
    FIXED: Улучшенная функция ожидания с детектированием ошибок NaN
    """
    start_time = asyncio.get_event_loop().time()
    last_progress = -1
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        if asyncio.get_event_loop().time() - start_time > timeout:
            raise Exception("Timeout waiting for generation")
        
        try:
            async with session.get(f"{comfy_url}/history/{prompt_id}") as resp:
                if resp.status == 200:
                    history = await resp.json()
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        
                        if status.get("completed", False):
                            return {"completed": True}
                        
                        # КРИТИЧЕСКИ ВАЖНО: Проверяем на ошибки
                        if "exception_message" in status:
                            error_msg = status["exception_message"]
                            print(f"[ERROR DETECTED] {error_msg}")
                            return {"completed": False, "error": error_msg}
                        
                        # Проверяем messages на ошибки
                        messages = status.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, list) and len(msg) >= 2:
                                msg_type = msg[0]
                                msg_content = msg[1]
                                if msg_type == "execution_error":
                                    error_text = str(msg_content)
                                    print(f"[EXECUTION ERROR] {error_text}")
                                    return {"completed": False, "error": error_text}
            
            # Получаем прогресс
            async with session.get(f"{comfy_url}/queue") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    running = data.get("queue_running", [])
                    
                    for item in running:
                        if item[1] == prompt_id:
                            # Извлекаем прогресс если есть
                            break
            
            consecutive_errors = 0
        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            consecutive_errors += 1
            print(f"Error checking status: {e}")
            if consecutive_errors >= max_consecutive_errors:
                raise Exception("ComfyUI port is not responding")
        except Exception:
            raise
        
        await asyncio.sleep(2)


async def get_output_file(prompt_id):
    """Находит самый новый выходной файл"""
    await asyncio.sleep(3)  # Даем время на сохранение файла
    
    try:
        files = []
        for filename in os.listdir(OUTPUT_DIR):
            # Ищем файлы с нужными префиксами
            if (filename.startswith(('wan_native_', 'wananimatev2_nat_', 'IAMCCS')) and 
                filename.endswith(('.mp4', '.gif', '.webm', '.avi', '.mov'))):
                filepath = os.path.join(OUTPUT_DIR, filename)
                files.append((filepath, os.path.getmtime(filepath)))
        
        if files:
            # Сортируем по времени модификации и берем самый новый
            files.sort(key=lambda x: x[1], reverse=True)
            return files[0][0]
        
        return None
    except Exception as e:
        print(f"Error getting output file: {e}")
        return None


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    if not check_access(message.from_user.id):
        await message.answer(
            "⛔ Access denied.\n\n"
            "Please contact admin to get access."
        )
        return
    
    status = await queue_manager.get_global_status()
    
    await message.answer(
        f"🎬 ComfyUI Motion Transfer Bot\n\n"
        f"📊 System Status:\n"
        f"Active GPUs: {status['total_ports']}\n"
        f"Processing: {status['processing_ports']}\n"
        f"Total queue: {status['total_queue']}\n\n"
        f"Choose an option:",
        reply_markup=get_main_keyboard()
    )


@dp.message(F.text == "New Generation")
async def button_new_generation(message: types.Message, state: FSMContext):
    if not check_access(message.from_user.id):
        return
    
    await message.answer(
        "Choose output resolution:",
        reply_markup=get_resolution_keyboard()
    )
    await state.set_state(GenerationStates.choosing_resolution)


@dp.message(F.text == "My Tasks")
async def button_my_tasks(message: types.Message):
    if not check_access(message.from_user.id):
        return
    
    tasks = await queue_manager.get_all_user_tasks(message.from_user.id)
    
    if not tasks:
        await message.answer("You have no active tasks")
        return
    
    text = "📋 Your tasks:\n\n"
    for i, task in enumerate(tasks, 1):
        status_emoji = {
            "waiting": "⏳",
            "processing": "⚙️",
            "completed": "✅",
            "error": "❌"
        }.get(task.status, "❓")
        
        text += f"{i}. {status_emoji} {task.status.upper()}\n"
        text += f"   Port: {task.port}\n"
        text += f"   Resolution: {task.width}x{task.height}\n"
        if task.retry_count > 0:
            text += f"   Retries: {task.retry_count}\n"
        text += f"   ID: {task.task_id[:8]}...\n\n"
    
    await message.answer(text)


@dp.message(F.text == "Queue Status")
async def button_queue_status(message: types.Message):
    if not check_access(message.from_user.id):
        return
    
    status = await queue_manager.get_global_status()
    
    text = "📊 System Status:\n\n"
    text += f"Active GPUs: {status['total_ports']}\n"
    text += f"Processing: {status['processing_ports']}\n"
    text += f"Total queue: {status['total_queue']}\n\n"
    
    text += "Port Status:\n"
    for port, port_info in status['port_status'].items():
        status_emoji = "⚙️" if port_info['is_processing'] else "✅"
        text += f"{status_emoji} Port {port}: "
        if port_info['is_processing']:
            text += "Processing"
        else:
            text += "Idle"
        text += f" (Queue: {port_info['queue_length']})\n"
    
    await message.answer(text)


@dp.message(GenerationStates.choosing_resolution, F.text.in_(list(RESOLUTIONS.keys())))
async def resolution_chosen(message: types.Message, state: FSMContext):
    if not check_access(message.from_user.id):
        return
    
    resolution = message.text
    width, height = RESOLUTIONS[resolution]
    
    await state.update_data(width=width, height=height)
    await message.answer(
        f"Resolution set: {width}x{height}\n\n"
        f"Now send your PHOTO (face image):",
        reply_markup=get_cancel_keyboard()
    )
    await state.set_state(GenerationStates.waiting_photo)


@dp.message(GenerationStates.choosing_resolution, F.text == "Cancel")
@dp.message(GenerationStates.waiting_photo, F.text == "Cancel")
@dp.message(GenerationStates.waiting_video, F.text == "Cancel")
async def cancel_generation(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "Generation cancelled.",
        reply_markup=get_main_keyboard()
    )


@dp.message(GenerationStates.waiting_photo, F.photo)
async def photo_received(message: types.Message, state: FSMContext):
    if not check_access(message.from_user.id):
        return
    
    photo = message.photo[-1]
    filename = f"photo_{message.from_user.id}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(INPUT_DIR, filename)
    
    await bot.download(photo, filepath)
    
    await state.update_data(photo_filename=filename)
    await message.answer(
        "✅ Photo received!\n\n"
        "Now send your VIDEO (motion source):",
        reply_markup=get_cancel_keyboard()
    )
    await state.set_state(GenerationStates.waiting_video)


@dp.message(GenerationStates.waiting_video, F.video)
async def video_received(message: types.Message, state: FSMContext):
    if not check_access(message.from_user.id):
        return
    
    video = message.video
    filename = f"video_{message.from_user.id}_{uuid.uuid4().hex[:8]}.mp4"
    filepath = os.path.join(INPUT_DIR, filename)
    
    await bot.download(video, filepath)
    
    data = await state.get_data()
    await state.update_data(video_filename=filename)
    
    # Создаем задачу
    task = QueueTask(
        task_id=uuid.uuid4().hex,
        user_id=message.from_user.id,
        username=message.from_user.username or str(message.from_user.id),
        photo_filename=data["photo_filename"],
        video_filename=filename,
        width=data["width"],
        height=data["height"]
    )
    
    try:
        # Добавляем задачу в очередь с ротацией портов
        port, position = await queue_manager.add_task(task)
        await state.clear()
        
        status = await queue_manager.get_global_status()
        
        await message.answer(
            f"✅ Task added to queue!\n\n"
            f"Port: {port}\n"
            f"Position in port queue: {position}\n"
            f"Resolution: {task.width}x{task.height}\n\n"
            f"System: {status['processing_ports']}/{status['total_ports']} GPUs active\n"
            f"Total queue: {status['total_queue']} tasks\n\n"
            f"You will be notified when processing starts.",
            reply_markup=get_main_keyboard()
        )
    except Exception as e:
        await message.answer(
            f"❌ Error adding task to queue:\n{str(e)}",
            reply_markup=get_main_keyboard()
        )
        await state.clear()
        cleanup_task_files(task)


@dp.message(Command("add"))
async def cmd_add_user(message: types.Message):
    if not is_admin(message.from_user.id):
        await message.answer("Admin only")
        return
    
    try:
        parts = message.text.split()
        if len(parts) != 2:
            await message.answer("Usage: /add USER_ID")
            return
        
        user_id = int(parts[1])
        whitelist.add(user_id)
        save_whitelist(whitelist)
        await message.answer(f"User {user_id} added")
    except ValueError:
        await message.answer("Invalid ID format")


@dp.message(Command("remove"))
async def cmd_remove_user(message: types.Message):
    if not is_admin(message.from_user.id):
        await message.answer("Admin only")
        return
    
    try:
        parts = message.text.split()
        if len(parts) != 2:
            await message.answer("Usage: /remove USER_ID")
            return
        
        user_id = int(parts[1])
        if user_id == ADMIN_ID:
            await message.answer("Cannot remove admin")
            return
        
        if user_id in whitelist:
            whitelist.remove(user_id)
            save_whitelist(whitelist)
            await message.answer(f"User {user_id} removed")
        else:
            await message.answer("User not found")
    except ValueError:
        await message.answer("Invalid ID format")


@dp.message(Command("list"))
async def cmd_list_users(message: types.Message):
    if not is_admin(message.from_user.id):
        await message.answer("Admin only")
        return
    
    users = "\n".join([f"- {uid}" for uid in sorted(whitelist)])
    await message.answer(f"Whitelist ({len(whitelist)}):\n\n{users}")


@dp.message(Command("myid"))
async def cmd_my_id(message: types.Message):
    await message.answer(f"Your ID: {message.from_user.id}")


@dp.message(Command("cancel"))
async def cmd_cancel(message: types.Message):
    if not check_access(message.from_user.id):
        return
    
    tasks = await queue_manager.get_all_user_tasks(message.from_user.id)
    
    if not tasks:
        await message.answer("No active tasks")
        return
    
    # Показываем кнопки для отмены каждой задачи
    keyboard = []
    for task in tasks[:10]:  # Показываем максимум 10 задач
        status_emoji = {
            "waiting": "⏳",
            "processing": "⚙️"
        }.get(task.status, "❓")
        
        button_text = f"{status_emoji} Port {task.port} - {task.status}"
        keyboard.append([InlineKeyboardButton(
            text=button_text,
            callback_data=f"cancel_{task.task_id}"
        )])
    
    markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
    await message.answer("Select task to cancel:", reply_markup=markup)


@dp.callback_query(F.data.startswith("cancel_"))
async def callback_cancel_task(callback: types.CallbackQuery):
    task_id = callback.data.replace("cancel_", "")
    
    success, status = await queue_manager.cancel_task(task_id, callback.from_user.id)
    
    if success:
        if status == "processing":
            # Прерываем обработку на сервере
            tasks = await queue_manager.get_all_user_tasks(callback.from_user.id)
            task = next((t for t in tasks if t.task_id == task_id), None)
            
            if task and task.port:
                comfy_url = port_manager.get_url_for_port(task.port)
                await interrupt_comfy_processing(comfy_url)
                await callback.answer("Task cancelled!")
        else:
            await callback.answer("Task removed from queue!")
    else:
        await callback.answer(status)
    
    await callback.message.delete()


@dp.message(Command("ports"))
async def cmd_ports(message: types.Message):
    """Показывает статус всех портов"""
    if not check_access(message.from_user.id):
        return
    
    active_ports = await port_manager.get_active_ports()
    
    text = "🔌 ComfyUI Ports Status:\n\n"
    
    for port in COMFY_PORTS:
        if port in active_ports:
            url = port_manager.get_url_for_port(port)
            port_queue = queue_manager.port_queues.get(port)
            
            if port_queue:
                queue_len = await port_queue.get_queue_length()
                is_busy = await port_queue.is_busy()
                
                status = "⚙️ Processing" if is_busy else "✅ Idle"
                text += f"✅ {port}: {status} (Queue: {queue_len})\n"
            else:
                text += f"✅ {port}: Active\n"
        else:
            text += f"❌ {port}: Not available\n"
    
    await message.answer(text)


@dp.message(Command("refresh_ports"))
async def cmd_refresh_ports(message: types.Message):
    """Перезагружает список активных портов (только для админа)"""
    if not is_admin(message.from_user.id):
        await message.answer("Admin only")
        return
    
    await message.answer("Refreshing ports...")
    
    active_ports = await port_manager.detect_active_ports()
    await queue_manager.initialize_queues()
    
    # Запускаем обработчики для новых портов
    for port in active_ports:
        port_queue = queue_manager.port_queues.get(port)
        if port_queue:
            asyncio.create_task(process_port_queue(port, port_queue))
    
    await message.answer(
        f"✅ Ports refreshed!\n\n"
        f"Active ports: {len(active_ports)}\n"
        f"Ports: {', '.join(map(str, active_ports))}"
    )


async def main():
    print("=" * 50)
    print("Multi-GPU ComfyUI Bot Starting... (FIXED VERSION)")
    print(f"Admin ID: {ADMIN_ID}")
    print(f"Whitelist: {len(whitelist)} users")
    print(f"Base Host: {COMFY_BASE_HOST}")
    print("=" * 50)
    
    # Определяем активные порты
    active_ports = await port_manager.detect_active_ports()
    
    if not active_ports:
        print("❌ ERROR: No active ComfyUI ports found!")
        print("Please make sure ComfyUI is running on one of these ports:")
        for port in COMFY_PORTS:
            print(f"  - {port}")
        return
    
    # Инициализируем очереди для всех портов
    await queue_manager.initialize_queues()
    
    # Запускаем обработчики для каждого порта
    for port in active_ports:
        port_queue = queue_manager.port_queues[port]
        asyncio.create_task(process_port_queue(port, port_queue))
        print(f"✓ Started queue processor for port {port}")
    
    print("=" * 50)
    print("✅ Bot is ready! (With NaN error handling and video duration fix)")
    print("=" * 50)
    
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())