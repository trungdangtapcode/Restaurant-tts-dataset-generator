# Restaurant TTS Dataset Generator

An automated pipeline to generate robust Vietnamese Text-to-Speech (TTS) datasets from raw JSON restaurant bills. This tool leverages the `valtec-tts` library and PyTorch, distributing the workload asynchronously across multiple GPUs to maximize performance.

## 🌟 Features

* **Smart Text Generaton**: Intelligently categorizes food and drinks. It automatically applies proper Vietnamese classifiers (e.g., `"1 chai 7Up"` or `"1 lon Pepsi"` for drinks, `"1 ổ Bánh mì"` for bakery, and `"1 phần/dĩa"` for normal food dishes).
* **Multi-GPU Multiprocessing**: Spawns independent TTS engine workers to specific GPUs to avoid VRAM overflow and maximize synthesis throughput.
* **Dry-Run Mode**: Allows you to simulate and validate the text phrasing rules instantly without initializing the heavy TTS GPU models.
* **Clean Architecture**: Modularized codebase making it highly readable and easy to maintain or expand with new linguistic rules.

---

## 📂 Project Structure

* `config.py` - Central configuration class (select GPUs, input JSON, output dir).
* `domain.py` - Core Data Classes modeling the JSON representation (`Bill`, `OrderItem`).
* `text_generator.py` - Linguistic rules engine enforcing sentence structure and dynamic vocabulary categorizations.
* `tts_worker.py` - The PyTorch & ValtecTTS wrapper, running asynchronously inside its isolated GPU boundary.
* `main.py` - The orchestrator bootstrapping `multiprocessing.Pool`, distributing tasks, and tracking progress with `tqdm`.

---

## 🚀 Installation & Setup

1. **Initialize Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

2. **Install Core Dependencies**
Make sure to install GPU-enabled PyTorch depending on your CUDA version (CUDA 12.1 is used across the A4000/A5000 instances).
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm "numpy<2.0.0"
```

3. **Install Valtec TTS Model Dependencies**
Since the `valtec-tts-repo` source code is now included directly in this project (fixing the missing `infer.py` issue on Kaggle/Colab), you just need to install it locally:
```bash
pip install ./valtec-tts-repo
```

---

## ⚙️ Configuration

Open `config.py` to modify your setup before generating the dataset:
```python
class Config:
    USE_GPUS = [0, 1]              # Define which GPU IDs to utilize (e.g., two RTX A4000)
    INPUT_FILE = "bills-training.json"  # Source JSON file
    OUTPUT_DIR = Path("dataset_output") # Directory to save .wav and .txt pairs
    VARIATIONS_PER_BILL = 3        # Number of unique spoken variations per bill
```

---

## 🏃‍♂️ Usage

### 1. Test Text Generation (Dry-Run)
Ensure your grammar routing and logic work perfectly without consuming any GPU resources:
```bash
python main.py --dry-run
```

### 2. Full TTS Synthesis
Spawn the multi-GPU workers to synthesize the audio end-to-end:
```bash
python main.py
```
*Note: You can safely press `Ctrl + C` at any time to gracefully terminate all child processes and free up GPU memory.*

---

## 📁 Output Format
The results will be dumped into your configured `OUTPUT_DIR` (default: `dataset_output/`). Each synthesized line creates a paired tuple:
* `[bill_id]_var[variation_id]_[speaker].wav`: The synthesized Vietnamese audio.
* `[bill_id]_var[variation_id]_[speaker].txt`: The exact transcript used, perfect for training Text-to-Speech models.
