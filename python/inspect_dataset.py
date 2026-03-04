import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from dataset import load_speech_commands_dataset, Config

def plot_spectrogram(spectrogram, ax):
    # Transpose so that the time is represented in the x-axis (columns).
    # Shape is (time, freq, 1) -> (freq, time)
    freq_bins = spectrogram.shape[1]
    time_bins = spectrogram.shape[0]
    
    # squeeze the channel dimension
    spec_2d = np.squeeze(spectrogram)
    
    # show origin='lower' to have 0 frequency at bottom
    X = np.arange(time_bins)
    Y = np.arange(freq_bins)
    ax.pcolormesh(X, Y, spec_2d.T, shading='auto')
    ax.set_ylabel('Mel Frequency Bins')
    ax.set_xlabel('Time Frames')

def main():
    config = Config()
    
    print("Preparing Datasets...")
    train_ds, val_ds, test_ds, target_names = load_speech_commands_dataset(config)
    
    print("Caching, shuffling, counting elements (might take a minute)...")
    
    # Counting elements (inefficient to do every time, but good for inspection)
    train_count = train_ds.reduce(0, lambda x, _: x + 1).numpy()
    val_count = val_ds.reduce(0, lambda x, _: x + 1).numpy()
    test_count = test_ds.reduce(0, lambda x, _: x + 1).numpy()
    
    print(f"\n--- Dataset Sizes ---")
    print(f"Train split size: {train_count}")
    print(f"Validation split size: {val_count}")
    print(f"Test split size: {test_count}")
    
    print("\n--- Tensor Shapes ---")
    for spectrogram, label in train_ds.take(1):
        print(f"Input Spectrogram Shape: {spectrogram.shape}")
        print(f"Input Spectrogram dtype: {spectrogram.dtype}")
        print(f"Label Shape: {label.shape}")
        print(f"Label dtype: {label.dtype}")
        print(f"Example mapped label value: {label.numpy()} -> {target_names[label.numpy()]}")

    print("\n--- Visualizing a Batch ---")
    
    # Batch the dataset to get a few examples easily
    batch_size = 6
    batched_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size)
    
    for spectrograms, labels in batched_ds.take(1):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Log-Mel Spectrogram Examples')
        
        for i in range(batch_size):
            r = i // 3
            c = i % 3
            ax = axes[r][c]
            spec = spectrograms[i].numpy()
            plot_spectrogram(spec, ax)
            
            label_name = target_names[labels[i].numpy()]
            ax.set_title(label_name)
        
        os.makedirs('plots', exist_ok=True)
        plot_path = os.path.join('plots', 'spectrogram_examples.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"\nSaved example spectrogram plots to {os.path.abspath(plot_path)}")
        break

if __name__ == '__main__':
    # Fix for memory growth if using GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    main()
