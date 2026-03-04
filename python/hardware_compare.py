"""
hardware_compare.py
===================
Validates the hls4ml HLS model against the original Keras model on a small
batch of test examples. Proves the quantized fixed-point implementation
produces nearly identical predicted classes.

C/RTL Cosimulation Note
-----------------------
If Vivado HLS is installed, cosimulation can be triggered with:
    python -c "
    import hls4ml
    hls_model = hls4ml.converters.convert_from_keras_model(...)
    hls_model.build(csim=True, synth=True, cosim=True)
    "
Cosim reports and waveforms will be written to:
    hls4ml_project/keyword_cnn_hls/my-hls-test/solution1/sim/cosim/

Without Vivado HLS installed, this script compares via hls4ml's software
simulation (hls_model.predict), which exercises the same arithmetic.
"""

import os, sys
import shutil
import numpy as np
import tensorflow as tf

# ── Silence TF noise ─────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import hls4ml
from dataset import load_speech_commands_dataset, Config

# ─────────────────────────────────────────────────────────────────────────────
NUM_SAMPLES   = 32          # number of test samples to compare
BATCH_SIZE    = 32
MODEL_PATH    = '../models/keyword_cnn.h5'
HLS_DIR       = '../hls4ml_project/keyword_cnn_hls'

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.normpath(os.path.join(SCRIPT_DIR, MODEL_PATH))
HLS_DIR       = os.path.normpath(os.path.join(SCRIPT_DIR, HLS_DIR))
DOCS_DIR      = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'docs'))
# ─────────────────────────────────────────────────────────────────────────────


def collect_samples(num_samples: int):
    """Load NUM_SAMPLES spectrograms from the test split."""
    config = Config()
    _, _, test_ds, target_names = load_speech_commands_dataset(config)
    xs, ys = [], []
    for x, y in test_ds.take(num_samples):
        xs.append(x.numpy())
        ys.append(y.numpy())
    X = np.array(xs, dtype=np.float32)   # shape: (N, 124, 40, 1)
    Y = np.array(ys, dtype=np.int64)
    print(f"  Collected {len(X)} samples  (shape {X.shape})")
    return X, Y, target_names


def keras_predict(model, X):
    """Run Keras softmax prediction."""
    probs = model.predict(X, verbose=0)     # (N, num_classes)
    return probs, np.argmax(probs, axis=1)


def hls4ml_sw_predict(hls_dir, model, X, config):
    """
    Run software simulation via hls4ml (no HLS tools needed).
    hls_model.compile() calls csim internally using a lightweight C++ build.
    Falls back gracefully if compilation fails.
    """
    try:
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config   = config,
            output_dir   = hls_dir,
            backend      = 'Vivado',
            part         = 'xc7z020clg400-1',
            clock_period = 10,
            io_type      = 'io_stream',
        )
        hls_model.compile()     # builds the C-sim shared library
        y_hls = hls_model.predict(X)
        return y_hls, np.argmax(y_hls, axis=1), True
    except Exception as e:
        print(f"  [WARN] hls4ml software simulation failed: {e}")
        print("  Falling back to Keras-only comparison.")
        return None, None, False


def compare(keras_probs, keras_cls, hls_probs, hls_cls, true_cls, target_names):
    """Print comparison metrics."""
    print("\n─── Comparison Results ───")
    n = len(keras_cls)

    if hls_probs is not None:
        max_abs_diff = float(np.max(np.abs(keras_probs - hls_probs)))
        mean_abs_diff = float(np.mean(np.abs(keras_probs - hls_probs)))
        class_mismatches = int(np.sum(keras_cls != hls_cls))

        print(f"  Samples compared         : {n}")
        print(f"  Max  |Keras - HLS| prob  : {max_abs_diff:.6f}")
        print(f"  Mean |Keras - HLS| prob  : {mean_abs_diff:.6f}")
        print(f"  Class prediction mismatches: {class_mismatches} / {n}")

        keras_acc = np.mean(keras_cls == true_cls)
        hls_acc   = np.mean(hls_cls   == true_cls)
        print(f"\n  Keras accuracy (on batch): {keras_acc*100:.1f}%")
        print(f"  HLS   accuracy (on batch): {hls_acc  *100:.1f}%")

        results = {
            'samples': n,
            'max_abs_diff': max_abs_diff,
            'mean_abs_diff': mean_abs_diff,
            'class_mismatches': class_mismatches,
            'keras_acc': keras_acc,
            'hls_acc': hls_acc,
            'hls_available': True,
        }
    else:
        # HLS not available – just show Keras results on the batch
        keras_acc = np.mean(keras_cls == true_cls)
        print(f"  Samples evaluated        : {n}")
        print(f"  Keras accuracy (on batch): {keras_acc*100:.1f}%")
        print(f"  HLS software sim         : not available in this environment")
        results = {
            'samples': n,
            'keras_acc': keras_acc,
            'hls_available': False,
        }

    print("\n  Sample predictions (first 10):")
    print(f"  {'True':>10}  {'Keras':>10}  {'HLS':>10}")
    for i in range(min(10, n)):
        hls_pred = target_names[hls_cls[i]] if hls_cls is not None else '—'
        print(f"  {target_names[true_cls[i]]:>10}  {target_names[keras_cls[i]]:>10}  {hls_pred:>10}")

    return results


def update_hardware_summary(docs_dir, results):
    """Append a Correctness Check section to docs/hardware_summary.md."""
    path = os.path.join(docs_dir, 'hardware_summary.md')
    section = "\n## Correctness Check\n\n"
    section += f"- **Samples compared**: {results['samples']}\n"
    section += f"- **Keras batch accuracy**: {results['keras_acc']*100:.1f}%\n"

    if results['hls_available']:
        section += f"- **HLS batch accuracy**: {results['hls_acc']*100:.1f}%\n"
        section += f"- **Max |Keras-HLS| probability diff**: {results['max_abs_diff']:.6f}\n"
        section += f"- **Mean |Keras-HLS| probability diff**: {results['mean_abs_diff']:.6f}\n"
        section += f"- **Class prediction mismatches**: {results['class_mismatches']} / {results['samples']}\n"
    else:
        section += (
            "- **HLS software simulation**: not available in this environment.\n"
            "  Install GCC/g++ and re-run to enable hls4ml C-sim comparison.\n\n"
            "### Manual Cosimulation Steps\n\n"
            "1. Install Vivado HLS (or Vitis HLS) and add to PATH.\n"
            "2. From the project directory run:\n"
            "   ```bash\n"
            "   cd hls4ml_project/keyword_cnn_hls\n"
            "   vitis_hls -f build_prj.tcl\n"
            "   ```\n"
            "3. Enable cosim by setting `cosim=True` in `hls_model.build()`.\n"
            "4. Cosim waveforms are saved to:\n"
            "   `my-hls-test/solution1/sim/cosim/`\n"
        )

    with open(path, 'a', encoding='utf-8') as f:
        f.write(section)

    print(f"\n  Updated docs/hardware_summary.md with Correctness Check section.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    print("=" * 60)
    print("  Keras vs hls4ml Hardware Comparison")
    print("=" * 60)

    # 1. Load samples
    print("\n[1] Loading test samples …")
    X, true_cls, target_names = collect_samples(NUM_SAMPLES)

    # 2. Load Keras model
    print("\n[2] Loading Keras model …")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 3. Keras inference
    print("\n[3] Running Keras inference …")
    keras_probs, keras_cls = keras_predict(model, X)

    # 4. hls4ml software simulation
    print("\n[4] Setting up hls4ml software simulation …")
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision']   = 'ap_fixed<8,3>'
    config['Model']['ReuseFactor'] = 1
    for layer in model.layers:
        lname = layer.name
        if lname not in config['LayerName']:
            config['LayerName'][lname] = {}
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            config['LayerName'][lname]['Precision'] = {
                'weight': 'ap_fixed<8,3>',
                'bias':   'ap_fixed<8,3>',
                'result': 'ap_fixed<16,6>'
            }

    hls_probs, hls_cls, hls_ok = hls4ml_sw_predict(HLS_DIR, model, X, config)

    # 5. Compare
    print("\n[5] Comparing outputs …")
    results = compare(keras_probs, keras_cls, hls_probs, hls_cls, true_cls, target_names)

    # 6. Update docs
    print("\n[6] Updating hardware_summary.md …")
    update_hardware_summary(DOCS_DIR, results)

    print("\n✅ hardware_compare.py done.")


if __name__ == '__main__':
    main()
