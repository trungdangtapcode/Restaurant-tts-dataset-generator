import fake_imp
import sys
import os
sys.path.insert(0, os.path.abspath("valtec-tts-repo"))
import fake_imp
import os
import sys
# HACK: valtec-tts' pip package is missing critical project files ('infer', 'src', etc.)
# We explicitly inject the cloned repo folder into sys.path to resolve ModuleNotFoundError.
repo_path = os.path.abspath("valtec-tts-repo")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import json
import random
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import time
import signal
import sys

# ==========================================
# USER CONFIGURATION
# ==========================================
class Config:
    # Select GPUs to use. Based on nvidia-smi, 2 A4000s are [0, 1]
    USE_GPUS = [0, 1]  
    INPUT_FILE = "bills-training.json"
    OUTPUT_DIR = Path("dataset_output")
    VARIATIONS_PER_BILL = 3

# ==========================================
# 1. DOMAIN MODELS (Data Structures)
# ==========================================
@dataclass
class OrderItem:
    name: str
    quantity: int

@dataclass
class Bill:
    bill_id: str
    table_number: int
    items: List[OrderItem]

    @classmethod
    def from_dict(cls, data: dict) -> 'Bill':
        items = [
            OrderItem(name=i.get("name", ""), quantity=i.get("quantity", 1))
            for i in data.get("items", [])
        ]
        return cls(
            bill_id=data.get("billId", "unknown"),
            table_number=data.get("tableNumber", 0),
            items=items
        )

# ==========================================
# 2. TEXT GENERATION (Rules & Logic)
# ==========================================
class BillSentenceFormatter:
    def format_quantity(self, item: OrderItem) -> str:
        # Keep speech phrases in Vietnamese as the dataset is in Vietnamese
        if item.quantity == 1:
            formats = ["", "1 ", "một ", "một phần ", "1 phần "]
        else:
            formats = [f"{item.quantity} ", f"{item.quantity} phần ", f"{item.quantity} dĩa ", f"{item.quantity} cái "]
        return random.choice(formats) + item.name

    def randomize_and_join_items(self, items: List[OrderItem]) -> str:
        if not items:
            return ""
        copied_items = list(items)
        random.shuffle(copied_items)
        phrases = [self.format_quantity(item) for item in copied_items]
        
        if len(phrases) == 1:
            return phrases[0]

        joined_str = ", ".join(phrases[:-1])
        connector = random.choice([" và ", ", với ", ", ", " thêm "])
        return f"{joined_str}{connector}{phrases[-1]}"

    def place_table_number(self, items_phrase: str, table_number: int) -> str:
        place_first = random.choice([True, False])
        if place_first:
            prefixes = [f"Bàn số {table_number}, ", f"Cho bàn {table_number}, ", f"Bàn {table_number} gọi ", f"Ghi cho bàn {table_number} ", f"Bàn {table_number} nà, "]
            return f"{random.choice(prefixes)}{items_phrase}"
        else:
            suffixes = [f" cho bàn số {table_number}", f" bàn {table_number} nhé", f", bàn {table_number}"]
            return f"{items_phrase}{random.choice(suffixes)}"

    def finalize_sentence(self, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence: return sentence
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith((".", "!", ",")):
            sentence += random.choice([".", " nhé.", " nha.", " nha em.", " nha quán."])
        return sentence

    def generate(self, bill: Bill) -> str:
        if not bill.items: return ""
        items_phrase = self.randomize_and_join_items(bill.items)
        sentence = self.place_table_number(items_phrase, bill.table_number)
        return self.finalize_sentence(sentence)


# ==========================================
# 3. TTS INFRASTRUCTURE (Worker Logic)
# ==========================================
class TTSEngineWorker:
    """TTS Worker independently initialized on each GPU."""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import lazily here due to `spawn` mechanism
        try:
            import torch
            from valtec_tts import TTS
            self.torch = torch
            self.TTS = TTS
        except ImportError:
            self.torch = None
            self.TTS = None
            print("Warning: valtec_tts or torch not installed. Running DUMMY mode.")
            
        if self.torch and self.torch.cuda.is_available():
            device_name = self.torch.cuda.get_device_name(0)
            print(f"[Worker] Initialized TTS on GPU: {device_name}")

        if self.TTS:
            self.tts = self.TTS()
            self.speakers = self.tts.list_speakers()
        else:
            self.tts = None
            self.speakers = ["DUMMY_SPEAKER"]

    def synthesize(self, text: str, bill_id: str, variation_id: int):
        speaker = random.choice(self.speakers)
        base_filename = f"{bill_id}_var{variation_id}_{speaker}"
        wav_path = self.output_dir / f"{base_filename}.wav"
        txt_path = self.output_dir / f"{base_filename}.txt"

        txt_path.write_text(text, encoding="utf-8")

        if self.tts:
            try:
                self.tts.speak(text, speaker=speaker, output_path=str(wav_path))
            except Exception as e:
                print(f"\n[Error] [{bill_id}] - {text}: {e}")
        else:
            time.sleep(0.01) # Dummy delay to test progress bar


# ==========================================
# 4. ORCHESTRATOR & MULTIPROCESSING
# ==========================================
_worker_engine = None
_formatter = BillSentenceFormatter()

def init_worker_queue(q, out_dir):
    """Worker initialization hook, must be defined at module level for Pickling."""
    signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore interrupt signal in worker
    gpu_id = q.get() # Retrieve assigned GPU_ID
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    global _worker_engine
    _worker_engine = TTSEngineWorker(out_dir)

def worker_task(task_data: Tuple[Bill, int]):
    """Main execution task for the worker."""
    bill, v_idx = task_data
    sentence = _formatter.generate(bill)
    if sentence:
        _worker_engine.synthesize(sentence, bill.bill_id, v_idx)
    return True

def load_bills(json_path: str) -> List[Bill]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [Bill.from_dict(b) for b in data]

def main():
    # Force PyTorch generic multiprocessing strategy constraint (spawn)
    mp.set_start_method('spawn', force=True)

    bills = load_bills(Config.INPUT_FILE)
    if not bills:
        print(f"Data not found in {Config.INPUT_FILE}")
        return

    tasks = []
    for bill in bills:
        for v_idx in range(Config.VARIATIONS_PER_BILL):
            tasks.append((bill, v_idx))

    print(f"==================================================")
    print(f"🎯 Starting massive TTS generation")
    print(f"   - Total bills           : {len(bills)}")
    print(f"   - Total outputs         : {len(tasks)}")
    print(f"   - Allocated GPUs        : {len(Config.USE_GPUS)} (GPUs {Config.USE_GPUS})")
    print(f"==================================================")

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    processes_count = len(Config.USE_GPUS)
    
    # Use MP Manager to safely proxy the Queue across spawns
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in Config.USE_GPUS:
        gpu_queue.put(gpu_id)

    pool = mp.Pool(
        processes=processes_count, 
        initializer=init_worker_queue, 
        initargs=(gpu_queue, Config.OUTPUT_DIR)
    )

    try:
        result_iter = pool.imap_unordered(worker_task, tasks)
        
        if has_tqdm:
            for _ in tqdm(result_iter, total=len(tasks), desc="Generation Progress"):
                pass
        else:
            completed = 0
            for _ in result_iter:
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"➜ Progress: {completed}/{len(tasks)} ({(completed/len(tasks))*100:.1f}%)", end='\r')
            print("\n")

        pool.close()
        pool.join()
        print("\n✅ DATASET GENERATION COMPLETED!")

    except KeyboardInterrupt:
        print("\n\n⛔ KILLED BY USER (SIGINT)!")
        pool.terminate()
        pool.join()
        print("Safely terminated all active background workers.")
        sys.exit(1)

if __name__ == "__main__":
    main()
