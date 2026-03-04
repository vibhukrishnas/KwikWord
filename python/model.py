from tensorflow.keras import layers, models

def get_tiny_cnn(input_shape, num_classes):
    """
    Creates a tiny Keras CNN optimized for hls4ml deployment.
    Uses basic operations: Conv2D, MaxPooling2D, ReLU, Dense.
    
    Args:
        input_shape: Shape of the input spectrogram tensor, e.g. (124, 40, 1)
        num_classes: Number of output commands
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(8, (3, 3), padding='same'),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Block 2
        layers.Conv2D(16, (3, 3), padding='same'),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense maps
        layers.Flatten(),
        layers.Dense(64),
        layers.ReLU(),
        layers.Dropout(0.3),  # Dropout helps generalization but isn't synthesized by HLS
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
