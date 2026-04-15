import os
import time
import random
from pathlib import Path
import sys

class TTSEngineWorker:
    """TTS Worker independently initialized on each GPU."""
    def __init__(self, output_dir: Path, dry_run: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        
        try:
            import torch
            import valtec_tts
            
            # --- AGGRESSIVE INFER MODULE HOTFIX ---
            # valtec_tts expects its own package directory to be in sys.path
            valtec_tts_path = os.path.abspath(os.path.dirname(valtec_tts.__file__))
            if valtec_tts_path not in sys.path:
                sys.path.insert(0, valtec_tts_path)
            
            # Additional fallback: manually resolve and import 'infer' ahead of time
            try:
                import infer
            except ImportError:
                import importlib.util
                infer_path = os.path.join(valtec_tts_path, 'infer.py')
                if os.path.exists(infer_path):
                    spec = importlib.util.spec_from_file_location("infer", infer_path)
                    infer_module = importlib.util.module_from_spec(spec)
                    sys.modules["infer"] = infer_module
                    spec.loader.exec_module(infer_module)
            # ----------------------------------------
            
            from valtec_tts import TTS
            self.torch = torch
            self.TTS = TTS
        except ImportError as e:
            self.torch = None
            self.TTS = None
            if not self.dry_run:
                print(f"Warning: Initialization error. Running DUMMY mode. Error: {e}")
            
        if not self.dry_run and self.torch and getattr(self.torch, 'cuda', None) and self.torch.cuda.is_available():
            device_name = self.torch.cuda.get_device_name(0)
            print(f"[Worker] Initialized TTS on GPU: {device_name}")

        if not self.dry_run and self.TTS:
            try:
                self.tts = self.TTS()
                self.speakers = self.tts.list_speakers()
            except Exception as e:
                print(f"Failed to load TTS model: {e}")
                self.tts = None
                self.speakers = ["DUMMY_SPEAKER"]
        else:
            self.tts = None
            self.speakers = ["SF", "NM1", "SM"]

    def synthesize(self, text: str, bill_id: str, variation_id: int):
        speaker = random.choice(self.speakers)
        base_filename = f"{bill_id}_var{variation_id}_{speaker}"
        wav_path = self.output_dir / f"{base_filename}.wav"
        txt_path = self.output_dir / f"{base_filename}.txt"

        txt_path.write_text(text, encoding="utf-8")

        if self.dry_run:
            print(f"[DRY_RUN] Text Generated: {text}")
            return

        if self.tts:
            try:
                self.tts.speak(text, speaker=speaker, output_path=str(wav_path))
            except Exception as e:
                print(f"\n[Error] [{bill_id}] - {text}: {e}")
        else:
            time.sleep(0.01)
