import os
import sys
import shutil
import json
import subprocess
import tensorflow as tf
import hls4ml


# ── Tool detection ──────────────────────────────────────────────────────────
def find_hls_tool():
    """Return the name of the first available HLS tool, or None."""
    for tool in ('vitis_hls', 'vivado_hls'):
        if shutil.which(tool):
            return tool
    return None


# ── Retry helper ────────────────────────────────────────────────────────────
def build_with_retry(hls_model, output_dir, config, model, tool):
    """
    Attempt hls_model.build() once.
    On failure, increase ReuseFactor (reduces LUT pressure) and retry once.
    Returns the hls4ml model that was successfully built, or raises.
    """
    print(f"  Using HLS tool : {tool}")
    try:
        print(f"  [Attempt 1] Running {tool} synthesis …")
        hls_model.build(csim=False, synth=True, export=False)
        return hls_model
    except Exception as e:
        print(f"\n  [Attempt 1 FAILED] Error:\n    {e}")
        
        # Simple fix: double the reuse factor to halve resource usage
        old_rf = config['Model']['ReuseFactor']
        new_rf = max(2, old_rf * 2)
        print(f"\n  ─── Auto-fix: ReuseFactor  {old_rf} → {new_rf} ───")
        config['Model']['ReuseFactor'] = new_rf

        retry_dir = output_dir + '_rf' + str(new_rf)
        print(f"  Regenerating project into: {retry_dir}")
        hls_model2 = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config   = config,
            output_dir   = retry_dir,
            backend      = 'Vivado',
            part         = 'xc7z020clg400-1',
            clock_period = 10,
            io_type      = 'io_stream',
        )
        hls_model2.write()

        try:
            print(f"\n  [Attempt 2] Running {tool} synthesis with RF={new_rf} …")
            hls_model2.build(csim=False, synth=True, export=False)
            return hls_model2
        except Exception as e2:
            print(f"\n  [Attempt 2 FAILED] Error:\n    {e2}")
            print("\n  ✖ Both synthesis attempts failed.")
            print("  Suggestions:")
            print("    1. Reduce model size (fewer filters).")
            print("    2. Try io_type='io_parallel' for smaller models.")
            print("    3. Check available BRAM/DSP on the target part.")
            raise RuntimeError("HLS build failed after retry") from e2


# ── Report parsing ──────────────────────────────────────────────────────────
def print_vivado_report(output_dir):
    """Attempt to read and summarise the Vivado HLS report."""
    try:
        report = hls4ml.report.read_vivado_report(output_dir)
        print("\n─── Vivado HLS Report ───")
        
        csynth = report.get('CSynthesisReport', {})
        timing = csynth.get('TimingReport', {})
        resources = csynth.get('AreaReport', {})
        perf = csynth.get('PerformanceReport', {})
        
        latency_min = perf.get('Latency', {}).get('LatencyMin', 'N/A')
        latency_max = perf.get('Latency', {}).get('LatencyMax', 'N/A')
        ii          = perf.get('Latency', {}).get('IntervalMin', 'N/A')
        
        res  = resources.get('Resources', {})
        avail = resources.get('AvailableResources', {})
        
        print(f"  Latency (min/max) : {latency_min} / {latency_max} clock cycles")
        print(f"  Init. Interval    : {ii} clock cycles")
        print(f"  LUT  : {res.get('LUT', 'N/A')}  / {avail.get('LUT', '?')}")
        print(f"  FF   : {res.get('FF', 'N/A')}  / {avail.get('FF', '?')}")
        print(f"  BRAM : {res.get('BRAM_18K', 'N/A')}  / {avail.get('BRAM_18K', '?')}")
        print(f"  DSP  : {res.get('DSP48E', 'N/A')}  / {avail.get('DSP48E', '?')}")
        
        return {
            'latency_min': latency_min, 'latency_max': latency_max, 'ii': ii,
            'LUT': res.get('LUT', 'N/A'), 'FF': res.get('FF', 'N/A'),
            'BRAM': res.get('BRAM_18K', 'N/A'), 'DSP': res.get('DSP48E', 'N/A'),
        }
    except Exception as e:
        print(f"\n  [WARN] Could not parse Vivado report: {e}")
        return {}


# ── Write hardware summary doc ───────────────────────────────────────────────
def write_hardware_summary(docs_dir, report_data, config, tool_used, synth_ran):
    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, 'hardware_summary.md')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Hardware Summary – Keyword CNN HLS Project\n\n")
        
        f.write("## Model & Target\n\n")
        f.write(f"- **Keras model**: `models/keyword_cnn.h5`\n")
        f.write(f"- **FPGA Part**: `xc7z020clg400-1` (Xilinx Zynq-7020 / PYNQ-Z2)\n")
        f.write(f"- **Clock Period**: 10 ns (100 MHz)\n")
        f.write(f"- **Backend**: Vivado HLS via hls4ml\n\n")
        
        f.write("## Quantization / HLS Config\n\n")
        f.write(f"- **Weight/Bias Precision**: `ap_fixed<8,3>` (8-bit signed fixed-point)\n")
        f.write(f"- **Accumulator Precision**: `ap_fixed<16,6>`\n")
        f.write(f"- **Reuse Factor**: `{config['Model']['ReuseFactor']}`\n")
        f.write(f"- **IO Type**: `io_stream` (pipelined streaming)\n\n")
        
        if synth_ran and report_data:
            f.write("## Synthesis Results\n\n")
            f.write(f"- **Latency (min)**: {report_data.get('latency_min', 'N/A')} clock cycles\n")
            f.write(f"- **Latency (max)**: {report_data.get('latency_max', 'N/A')} clock cycles\n")
            f.write(f"- **Initiation Interval**: {report_data.get('ii', 'N/A')} clock cycles\n")
            f.write(f"- **LUT usage**: {report_data.get('LUT', 'N/A')}\n")
            f.write(f"- **FF usage**: {report_data.get('FF', 'N/A')}\n")
            f.write(f"- **BRAM usage**: {report_data.get('BRAM', 'N/A')}\n")
            f.write(f"- **DSP usage**: {report_data.get('DSP', 'N/A')}\n")
        else:
            f.write("## Synthesis Results\n\n")
            f.write("> ⚠️ Synthesis was not run — Vivado/Vitis HLS was not found on PATH.\n")
            f.write("> Run `vitis_hls -f build_prj.tcl` from inside the project directory.\n\n")
            f.write("## How to Synthesize\n\n")
            f.write("```bash\n")
            f.write("cd hls4ml_project/keyword_cnn_hls\n")
            f.write("vitis_hls -f build_prj.tcl   # or vivado_hls -f build_prj.tcl\n")
            f.write("```\n\n")
            f.write("After synthesis, the RTL will be in:\n")
            f.write("`hls4ml_project/keyword_cnn_hls/my-hls-test/solution1/syn/verilog/`\n")
    
    print(f"\n📄 Hardware summary written to: {path}")
    return path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(os.path.join(script_dir, '..', 'models', 'keyword_cnn.h5'))
    output_dir = os.path.normpath(os.path.join(script_dir, '..', 'hls4ml_project', 'keyword_cnn_hls'))
    docs_dir   = os.path.normpath(os.path.join(script_dir, '..', 'docs'))

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"  Input shape : {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Params      : {model.count_params():,}")

    # ── Build hls4ml config ───────────────────────────────────────
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision']   = 'ap_fixed<8,3>'
    config['Model']['ReuseFactor'] = 1

    for layer in model.layers:
        lname = layer.name
        if lname not in config['LayerName']:
            config['LayerName'][lname] = {}
        lcfg = config['LayerName'][lname]
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            lcfg['Precision'] = {
                'weight': 'ap_fixed<8,3>',
                'bias':   'ap_fixed<8,3>',
                'result': 'ap_fixed<16,6>'
            }

    # ── Print config summary ──────────────────────────────────────
    print("\n─── hls4ml Configuration ───")
    print(f"  Backend         : Vivado")
    print(f"  Part            : xc7z020clg400-1")
    print(f"  Clock           : 10 ns")
    print(f"  Precision       : {config['Model']['Precision']}")
    print(f"  Reuse Factor    : {config['Model']['ReuseFactor']}")
    print(f"  IO Type         : io_stream")
    print(f"  Output dir      : {output_dir}")

    # ── Convert (generate project files) ─────────────────────────
    print("\nGenerating HLS project files …")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config   = config,
        output_dir   = output_dir,
        backend      = 'Vivado',
        part         = 'xc7z020clg400-1',
        clock_period = 10,
        io_type      = 'io_stream',
    )
    hls_model.write()
    print(f"  ✅ Project files written to {output_dir}")

    # ── Detect HLS tool & optionally synthesize ───────────────────
    tool = find_hls_tool()
    synth_ran   = False
    report_data = {}

    if tool:
        print(f"\n✅ HLS tool detected: {tool}")
        try:
            built_model = build_with_retry(hls_model, output_dir, config, model, tool)
            synth_ran   = True
            report_data = print_vivado_report(output_dir)
        except RuntimeError:
            print("\n  Skipping report — build did not succeed.")
    else:
        print(
            "\n⚠️  Neither vitis_hls nor vivado_hls is found on PATH.\n"
            "   HLS project files have been written successfully.\n"
            "   To run synthesis manually:\n"
            f"     cd {output_dir}\n"
            "     vitis_hls -f build_prj.tcl"
        )

    # ── Write hardware summary ────────────────────────────────────
    write_hardware_summary(docs_dir, report_data, config, tool, synth_ran)

    # ── Final directory listing ───────────────────────────────────
    print("\n─── Key Generated Files ───")
    for fname in ['firmware/myproject.cpp', 'firmware/myproject.h',
                  'firmware/parameters.h', 'build_prj.tcl', 'hls4ml_config.yml']:
        fpath = os.path.join(output_dir, fname)
        exists = '✅' if os.path.exists(fpath) else '❌'
        print(f"  {exists}  {fname}")


if __name__ == '__main__':
    main()
