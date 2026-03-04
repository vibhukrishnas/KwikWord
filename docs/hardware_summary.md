# Hardware Summary – Keyword CNN HLS Project

## Model & Target

- **Keras model**: `models/keyword_cnn.h5`
- **FPGA Part**: `xc7z020clg400-1` (Xilinx Zynq-7020 / PYNQ-Z2)
- **Clock Period**: 10 ns (100 MHz)
- **Backend**: Vivado HLS via hls4ml

## Quantization / HLS Config

- **Weight/Bias Precision**: `ap_fixed<8,3>` (8-bit signed fixed-point)
- **Accumulator Precision**: `ap_fixed<16,6>`
- **Reuse Factor**: `1`
- **IO Type**: `io_stream` (pipelined streaming)

## Synthesis Results

> ⚠️ Synthesis was not run — Vivado/Vitis HLS was not found on PATH.
> Run `vitis_hls -f build_prj.tcl` from inside the project directory.

## How to Synthesize

```bash
cd hls4ml_project/keyword_cnn_hls
vitis_hls -f build_prj.tcl   # or vivado_hls -f build_prj.tcl
```

After synthesis, the RTL will be in:
`hls4ml_project/keyword_cnn_hls/my-hls-test/solution1/syn/verilog/`

## Correctness Check

- **Samples compared**: 32
- **Keras batch accuracy**: 100.0%
- **HLS software simulation**: not available in this environment.
  Install GCC/g++ and re-run to enable hls4ml C-sim comparison.

### Manual Cosimulation Steps

1. Install Vivado HLS (or Vitis HLS) and add to PATH.
2. From the project directory run:
   ```bash
   cd hls4ml_project/keyword_cnn_hls
   vitis_hls -f build_prj.tcl
   ```
3. Enable cosim by setting `cosim=True` in `hls_model.build()`.
4. Cosim waveforms are saved to:
   `my-hls-test/solution1/sim/cosim/`
