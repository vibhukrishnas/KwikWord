import tensorflow as tf
import numpy as np
import os

class Config:
    def __init__(self):
        self.sample_rate = 16000
        # Only use a subset of commands for this tiny dataset
        self.target_commands = ["yes", "no", "up", "down", "stop", "go"]
        
        # Spectrogram parameters
        self.frame_length = 255
        self.frame_step = 128
        self.fft_length = 256
        
        # Mel spectrogram parameters
        self.num_mel_bins = 40
        self.lower_edge_hertz = 20.0
        self.upper_edge_hertz = 4000.0

def get_spectrogram(audio, config):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([config.sample_rate] - tf.shape(audio), dtype=tf.float32)
    
    # Concatenate audio with padding so that all audio clips will be of the same length
    audio = tf.cast(audio, tf.float32)
    equal_length = tf.concat([audio, zero_padding], 0)
    
    # Compute STFT
    spectrogram = tf.signal.stft(
        equal_length, frame_length=config.frame_length, frame_step=config.frame_step, fft_length=config.fft_length)
    
    # Abs
    spectrogram = tf.abs(spectrogram)
    
    return spectrogram

def get_log_mel_spectrogram(audio, config):
    spectrogram = get_spectrogram(audio, config)
    
    # Instantiate the Mel weight matrix
    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        config.num_mel_bins,
        num_spectrogram_bins,
        config.sample_rate,
        config.lower_edge_hertz,
        config.upper_edge_hertz)
    
    # Apply Mel weight matrix
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    
    # Compute log Mel spectrogram
    # Add a small offset to prevent taking the log of zero
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    # Expand dims so that it has channel dimension [time, mel_bins, 1]
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)
    
    return log_mel_spectrogram

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path, target_names_tensor):
    # file_path is like .../data/speech_commands_v0.02/yes/00f0204f_nohash_0.wav
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    
    # target_names_tensor is a 1D string tensor. We find the index where label == target
    matches = tf.equal(target_names_tensor, label)
    index = tf.argmax(tf.cast(matches, tf.int64))
    return index

def load_speech_commands_dataset(config: Config):
    """
    Loads and preprocesses the speech_commands dataset using tf.io rather than TFDS.
    Filters only the commands specified in config.target_commands.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'speech_commands_v0.02'))
    
    if not os.path.exists(os.path.join(data_dir, 'yes')):
        print("Downloading dataset using tf.keras.utils.get_file...")
        tf.keras.utils.get_file(
            'speech_commands_v0.02.tar.gz',
            origin="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
            extract=True,
            cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            cache_subdir='data/speech_commands_v0.02'
        )

    print(f"Loading files from: {data_dir}")
    print(f"Target commands: {config.target_commands}")
    
    filenames = []
    for cmd in config.target_commands:
        cmd_dir = os.path.join(data_dir, cmd)
        paths = tf.io.gfile.glob(str(cmd_dir) + '/*.wav')
        filenames.extend(paths)
        
    print(f"Found {len(filenames)} files.")
        
    filenames = tf.random.shuffle(tf.constant(filenames))
    
    # Split: 80% train, 10% val, 10% test
    num_samples = len(filenames)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    
    target_names_tensor = tf.constant(config.target_commands)

    train_files = filenames[:train_size]
    val_files = filenames[train_size: train_size + val_size]
    test_files = filenames[train_size + val_size:]

    def process_path(file_path):
        label = get_label(file_path, target_names_tensor)
        audio_binary = tf.io.read_file(file_path)
        audio = decode_audio(audio_binary)
        log_mel_spec = get_log_mel_spectrogram(audio, config)
        return log_mel_spec, label

    train_ds = tf.data.Dataset.from_tensor_slices(train_files).map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(val_files).map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices(test_files).map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, config.target_commands
