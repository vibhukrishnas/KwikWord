# Edge AI FPGA Keyword Spotter

A minimal end‑to‑end pipeline for **keyword spotting** on an FPGA using the Google Speech Commands v2 dataset, TensorFlow/Keras, and [hls4ml](https://fastmachinelearning.org/hls4ml/).

---

## What This Project Does

1. **Loads** a subset of the [Google Speech Commands v2](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) dataset (`yes / no / up / down / stop / go`).
2. **Preprocesses** each 1‑second audio clip into a log‑Mel spectrogram `(124, 40, 1)`.
3. **Trains** a tiny 3‑block CNN (~160 K parameters) optimized for FPGA synthesis.
4. **Evaluates** the trained model on the held‑out test set.
5. **Converts** the Keras model to synthesisable C++ HLS code via hls4ml, targeting Xilinx Vivado/Vitis HLS.
6. **Validates** that the HLS model produces the same predictions as the Keras model.

---

## Project Structure

```
.
├── python/
│   ├── requirements.txt        # Python dependencies
│   ├── dataset.py              # Data loading & log-Mel preprocessing
│   ├── inspect_dataset.py      # Sanity-check: shapes, counts, spectrogram plots
│   ├── model.py                # Tiny Keras CNN definition
│   ├── train.py                # Training script (saves models/keyword_cnn.h5)
│   ├── eval.py                 # Test-set evaluation & confusion matrix
│   ├── hls4ml_export.py        # Convert model → Vivado HLS project
│   └── hardware_compare.py     # Compare Keras vs HLS predictions
├── models/
│   └── keyword_cnn.h5          # Best saved Keras model (after training)
├── data/
│   └── speech_commands_v0.02/  # Downloaded dataset (auto-created on first run)
├── hls4ml_project/
│   └── keyword_cnn_hls/        # Generated HLS C++ project (ready for Vivado)
├── docs/
│   └── hardware_summary.md     # HLS config, resource estimates, correctness check
├── setup_commands.txt          # First-time setup commands
└── README.md
```

---

## Prerequisites

| Tool | Required | Install |
|---|---|---|
| Python 3.10+ | ✅ | [python.org](https://www.python.org) |
| TensorFlow | ✅ | `pip install tensorflow` |
| hls4ml | ✅ | `pip install hls4ml` |
| numpy / matplotlib | ✅ | `pip install numpy matplotlib` |
| tensorflow-datasets | ✅ | `pip install tensorflow-datasets` |
| pydub + audioop-lts | ✅ | `pip install pydub audioop-lts` |
| Vivado HLS / Vitis HLS | Optional | [Xilinx download](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-suite.html) |
| GCC / g++ | Optional | Required for hls4ml C-sim |

Install all Python dependencies at once:
```powershell
pip install -r python/requirements.txt
```

---

## Step‑by‑Step Usage

### 1 · Inspect the Dataset
Downloads and caches the dataset automatically on first run.
```powershell
cd python
python inspect_dataset.py
```
Outputs tensor shapes, split sizes, and saves example spectrogram images to `python/plots/`.

### 2 · Train the CNN
```powershell
python train.py
```
Trains for up to 20 epochs with early stopping. Saves the best model to `models/keyword_cnn.h5`.

**Results achieved**:
- Test accuracy: **93.5%**
- Parameters: **159,942**
- Input shape: `(124, 40, 1)` — log-Mel spectrogram

### 3 · Evaluate on Test Set
```powershell
python eval.py
```
Prints test accuracy and a full 6×6 confusion matrix.

### 4 · Export to Vivado HLS
```powershell
python hls4ml_export.py
```
- Quantizes weights to `ap_fixed<8,3>` (8-bit fixed-point).
- Generates the full C++ HLS project in `hls4ml_project/keyword_cnn_hls/`.
- If Vivado/Vitis HLS is on PATH, it also runs synthesis and prints resource estimates.

### 5 · Validate Hardware vs Software
```powershell
python hardware_compare.py
```
Compares Keras vs hls4ml predictions on 32 test samples. Reports max absolute output difference and class mismatch count.

### 6 · Run RTL Synthesis (requires Vivado HLS)
```powershell
cd hls4ml_project/keyword_cnn_hls
vitis_hls -f build_prj.tcl
# or:  vivado_hls -f build_prj.tcl
```
Generated Verilog RTL will appear in:
`keyword_cnn_hls/my-hls-test/solution1/syn/verilog/`

---

## Model Architecture

```
Input (124, 40, 1)
  │
  ├─ Conv2D(8, 3×3) → ReLU → MaxPool(2×2)
  ├─ Conv2D(16, 3×3) → ReLU → MaxPool(2×2)
  ├─ Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
  ├─ Flatten
  ├─ Dense(64) → ReLU → Dropout(0.3)
  └─ Dense(6) → Softmax
```
Only FPGA-friendly ops are used. Dropout is ignored at inference by hls4ml.

---

## HLS Configuration

| Setting | Value |
|---|---|
| Backend | Vivado |
| FPGA Part | `xc7z020clg400-1` (PYNQ‑Z2) |
| Clock | 10 ns / 100 MHz |
| Weight Precision | `ap_fixed<8,3>` |
| Accumulator Precision | `ap_fixed<16,6>` |
| Reuse Factor | 1 (max throughput) |
| IO Type | `io_stream` |

---

## Workflow Diagram

```
Speech Commands WAV files
        │
   dataset.py  →  log-Mel spectrogram (124×40×1)
        │
   train.py   →  keyword_cnn.h5  (Keras, 93.5% accuracy)
        │
   hls4ml_export.py  →  keyword_cnn_hls/  (C++ HLS source)
        │
   [Vivado/Vitis HLS]  →  Verilog RTL + timing/resource report
```
