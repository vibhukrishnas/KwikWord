# Round 1 Report – Edge AI FPGA Keyword Spotter

> Date: 2026-03-04 | Target: Xilinx Zynq-7020 (PYNQ-Z2)

## Objective

Build a minimal, working pipeline that trains a tiny CNN for keyword spotting on the Google Speech Commands v2 dataset, converts it to fixed-point HLS C++ via hls4ml, and generates an FPGA-ready Vivado/Vitis HLS project.

## Pipeline Status Summary

| Stage | Name | Status |
|---|---|---|
| 1 | Workspace setup | RUNS_SUCCESSFULLY |
| 2 | Dataset + preprocessing | RUNS_SUCCESSFULLY |
| 3 | Model + training + eval | RUNS_SUCCESSFULLY |
| 4 | hls4ml export | RUNS_SUCCESSFULLY |
| 5 | HLS build + RTL + report | BLOCKED_BY_ENV |
| 6 | Hardware vs Keras check | RUNS_SUCCESSFULLY |
| 7 | Round-1 report | RUNS_SUCCESSFULLY |

**5 of 7 stages** run successfully end-to-end. Stage 5 is blocked only by the absence of Xilinx Vivado/Vitis HLS on the current machine.

## Key Metrics

| Metric | Value |
|---|---|
| Target commands | yes, no, up, down, stop, go |
| Dataset size | 23,377 clips (6 classes) |
| Train / Val / Test split | 18,701 / 2,337 / 2,339 |
| Input tensor shape | (124, 40, 1) log-Mel spectrogram |
| Model parameters | 159,942 |
| Test accuracy (20 epochs) | 93.50% |
| HLS weight precision | ap_fixed<8,3> (8-bit) |
| HLS accumulator precision | ap_fixed<16,6> |
| FPGA part | xc7z020clg400-1 |
| Clock target | 100 MHz (10 ns) |
| HLS project files generated | 104 files |

## Architecture

```
Input (124, 40, 1)
  Conv2D(8, 3x3) -> ReLU -> MaxPool(2x2)
  Conv2D(16, 3x3) -> ReLU -> MaxPool(2x2)
  Conv2D(32, 3x3) -> ReLU -> MaxPool(2x2)
  Flatten -> Dense(64) -> ReLU -> Dropout(0.3)
  Dense(6) -> Softmax
```

All layers are hls4ml-compatible. Dropout is ignored at inference.

## Known Limitations

- **No Vivado/Vitis HLS** on current machine — RTL synthesis and resource reports are pending.
- **No GCC/g++** on PATH — hls4ml C-simulation (`hls_model.compile()`) cannot run; Keras-only comparison was used.
- **Hardest confusion pair**: `go` vs `no` — phonetically similar, accounts for most misclassifications.

## Files Delivered

| File | Purpose |
|---|---|
| `python/dataset.py` | Data loader + log-Mel preprocessing |
| `python/inspect_dataset.py` | Shape/count inspection + spectrogram plots |
| `python/model.py` | Tiny CNN definition |
| `python/train.py` | Training with early stopping (supports `--epochs N`) |
| `python/eval.py` | Test accuracy + confusion matrix |
| `python/hls4ml_export.py` | Keras -> HLS conversion with auto-retry |
| `python/hardware_compare.py` | Keras vs HLS prediction comparison |
| `models/keyword_cnn.h5` | Trained model weights |
| `hls4ml_project/keyword_cnn_hls/` | Generated HLS C++ project |
| `docs/hardware_summary.md` | HLS config + correctness check |
| `docs/pipeline_status.md` | QA audit results |
| `docs/round1_report.md` | This report |

## Next Steps (Round 2)

1. Install Vivado/Vitis HLS and run `vitis_hls -f build_prj.tcl` to synthesize RTL.
2. Install MinGW-w64 (g++) to enable hls4ml C-simulation for quantitative Keras-vs-HLS comparison.
3. Review synthesis resource report (LUT, FF, BRAM, DSP) and iterate on ReuseFactor if needed.
4. Deploy to PYNQ-Z2 board for real-time inference demo.
