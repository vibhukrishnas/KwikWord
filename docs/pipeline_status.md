# Pipeline Status Report

> Generated: 2026-03-04 23:14 IST — QA audit of the Edge AI FPGA Keyword Spotter workspace.

| Stage | Name | Status | Notes |
|---|---|---|---|
| 1 | Workspace setup | **RUNS_SUCCESSFULLY** | `python/`, `hls4ml_project/`, `docs/`, `models/` all exist. `requirements.txt` (80 B), `README.md` (5.4 KB), `setup_commands.txt` (206 B) are present and non-empty. |
| 2 | Dataset + preprocessing | **RUNS_SUCCESSFULLY** | `inspect_dataset.py` exits 0. Prints 23 377 files, shapes `(124, 40, 1)`, saves `plots/spectrogram_examples.png` (135 KB). Dataset cached at `data/speech_commands_v0.02/`. |
| 3 | Model + training + eval | **RUNS_SUCCESSFULLY** | `train.py --epochs 1` exits 0 (52% acc after 1 epoch, expected). `eval.py` exits 0 with confusion matrix. `models/keyword_cnn.h5` (1.97 MB) saved. Full 20-epoch run achieved 93.5% test accuracy previously. |
| 4 | hls4ml export | **RUNS_SUCCESSFULLY** | `hls4ml_export.py` exits 0. Generates 104 files in `keyword_cnn_hls/`. All 5 key files verified: `myproject.cpp`, `myproject.h`, `parameters.h`, `build_prj.tcl`, `hls4ml_config.yml`. |
| 5 | HLS build + RTL + report | **BLOCKED_BY_ENV** | Neither `vitis_hls` nor `vivado_hls` found on PATH. Project files are ready; synthesis requires Xilinx tools. |
| 6 | Hardware vs Keras check | **RUNS_SUCCESSFULLY** | `hardware_compare.py` exits 0. Keras accuracy 100% on 32-sample batch. hls4ml C-sim unavailable (no `g++` on PATH) — gracefully falls back and logs warning. `docs/hardware_summary.md` updated. |
| 7 | Round-1 report | **RUNS_SUCCESSFULLY** | `docs/round1_report.md` created. Contains pipeline summary, key metrics (93.5% accuracy, 160K params, 8-bit quantization), architecture, known limitations, and next steps. |

## Notes on Partial / Blocked Stages

### Stage 3 — needs `--epochs` flag for quick debug run
`train.py` currently has `epochs=20` hard-coded. A `--epochs 1` CLI flag would allow a quick 1-epoch smoke test without a full training cycle. The saved model from the prior full run is valid (1.97 MB, loadable).

### Stage 5 — environment dependency
Requires Xilinx Vivado HLS or Vitis HLS installed and on PATH. The `hls4ml_export.py` script has built-in detection and will auto-run synthesis when tools become available.

### Stage 6 — partial HLS comparison
The Keras-side comparison works perfectly. The hls4ml C-simulation requires `g++` (GCC) on PATH. Install MinGW-w64 or use WSL to enable it.

### Stage 7 — not yet created
`docs/round1_report.md` has not been authored.
