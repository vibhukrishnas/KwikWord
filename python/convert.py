import os
import tensorflow as tf
import hls4ml

def main():
    model_path = 'models/kws_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train.py first.")
        return

    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print("Creating hls4ml configuration...")
    # Generate default configuration
    config = hls4ml.utils.config_from_keras_model(model, granularity='Model')
    
    # Tweak settings for FPGA
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    config['Model']['ReuseFactor'] = 1
    
    output_dir = '../hls4ml_project/kws_hls'
    
    print("Converting Keras model to hls4ml project...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        part='xc7z020clg400-1'  # Example part (e.g., PYNQ-Z2)
    )

    print("Compiling HLS model in software to verify...")
    hls_model.compile()

    print(f"HLS project successfully generated at {output_dir}")
    print("You can now open the project in Vivado HLS / Vitis HLS and synthesize it.")

if __name__ == '__main__':
    main()
