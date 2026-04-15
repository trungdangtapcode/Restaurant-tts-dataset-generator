import os
import time
import random
from pathlib import Path
import sys
import subprocess

class TTSEngineWorker:
    """TTS Worker independently initialized on each GPU."""
    def __init__(self, output_dir: Path, dry_run: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        
        try:
            import sys
            import types
            if 'imp' not in sys.modules:
                sys.modules['imp'] = types.ModuleType('imp')
                
            import torch
            
            # =========================================================
            # THE REAL FIX: `pip install` DOES NOT INCLUDE `infer.py` AND `src/`!
            # Since you already added `valtec-tts-repo` to your project folder,
            # we simply add it to sys.path so it can find things properly.
            # =========================================================
            base_dir = os.path.dirname(os.path.abspath(__file__))
            valtec_src_dir = os.path.join(base_dir, "valtec-tts-repo")
            
            if not os.path.exists(os.path.join(valtec_src_dir, "infer.py")):
                if not self.dry_run:
                    print(f"Warning: valtec-tts-repo not found at {valtec_src_dir}")
            
            # Prioritize our pure source directory
            if valtec_src_dir not in sys.path:
                sys.path.insert(0, valtec_src_dir)
            
            # Explicitly import infer before TTS to guarantee success
            import infer
            from valtec_tts import TTS
            
            self.torch = torch
            self.TTS = TTS
        except ImportError as e:
            self.torch = None
            self.TTS = None
            if not self.dry_run:
                print(f"Warning: Initialization error. Running DUMMY mode. Error: {e}")
            
        if not self.dry_run and getattr(self, 'torch', None) and getattr(self.torch, 'cuda', None) and self.torch.cuda.is_available():
            device_name = self.torch.cuda.get_device_name(0)
            print(f"[Worker] Initialized TTS on GPU: {device_name}")

        if not self.dry_run and self.TTS:
            try:
                # Force current working directory briefly to help internal paths in valtec
                original_cwd = os.getcwd()
                os.chdir(valtec_src_dir)
                
                self.tts = self.TTS()
                self.speakers = self.tts.list_speakers()
                
                os.chdir(original_cwd)
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

        if self.dry_run:
            print(f"[DRY_RUN] Text Generated: {text}")
            return
            
        # In dummy/mock mode (failed init), DO NOT write any files to mask errors
        if not self.tts:
            time.sleep(0.01)
            return

        txt_path.write_text(text, encoding="utf-8")

        try:
            self.tts.speak(text, speaker=speaker, output_path=str(wav_path))
        except Exception as e:
            print(f"\n[Error/Skip] [{bill_id}] - {text}: {e}")

