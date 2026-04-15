import os
import json
import signal
import sys
import argparse
import multiprocessing as mp
from typing import List, Tuple

from config import Config
from domain import Bill
from text_generator import BillSentenceFormatter
from tts_worker import TTSEngineWorker

_worker_engine = None
_formatter = BillSentenceFormatter()

def init_worker_queue(q, out_dir, dry_run):
    """Worker initialization hook, must be defined at module level for Pickling."""
    signal.signal(signal.SIGINT, signal.SIG_IGN) 
    gpu_id = q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    global _worker_engine
    _worker_engine = TTSEngineWorker(out_dir, dry_run=dry_run)

def worker_task(task_data: Tuple[Bill, int]):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Test text generation without loading TTS models/GPU')
    args = parser.parse_args()

    # Disable MP if dry-run to test safely in one thread
    if args.dry_run:
        print("--- RUNNING DRY-RUN MODE (TEXT GENERATION TEST ONLY) ---")
        bills = load_bills(Config.INPUT_FILE)
        global _worker_engine
        _worker_engine = TTSEngineWorker(Config.OUTPUT_DIR, dry_run=True)
        count = 0
        for bill in bills[:5]:  # Test 5 bills
            for v_idx in range(2):
                worker_task((bill, v_idx))
                count += 1
        print(f"--- FISHED DRY-RUN ({count} samples) ---")
        sys.exit(0)

    mp.set_start_method('spawn', force=True)
    bills = load_bills(Config.INPUT_FILE)
    if not bills:
        return

    tasks = [(bill, v_idx) for bill in bills for v_idx in range(Config.VARIATIONS_PER_BILL)]

    print(f"==================================================")
    print(f"🎯 Starting massive TTS generation")
    print(f"   - Total outputs         : {len(tasks)}")
    print(f"==================================================")

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    manager = mp.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in Config.USE_GPUS:
        gpu_queue.put(gpu_id)

    pool = mp.Pool(
        processes=len(Config.USE_GPUS), 
        initializer=init_worker_queue, 
        initargs=(gpu_queue, Config.OUTPUT_DIR, False)
    )

    try:
        result_iter = pool.imap_unordered(worker_task, tasks)
        if has_tqdm:
            for _ in tqdm(result_iter, total=len(tasks), desc="Generation Progress"):
                pass
        else:
            for i, _ in enumerate(result_iter):
                if (i+1) % 10 == 0:
                    print(f"➜ Progress: {i+1}/{len(tasks)}", end='\r')
            print("\n")

        pool.close()
        pool.join()
        print("\n✅ DATASET GENERATION COMPLETED!")

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        sys.exit(1)

if __name__ == "__main__":
    main()
