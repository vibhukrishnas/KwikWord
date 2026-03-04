import os
import argparse
import tensorflow as tf
from dataset import load_speech_commands_dataset, Config
from model import get_tiny_cnn

def main(epochs=20):
    config = Config()
    
    # 1. Load Dataset
    print("Loading Datasets...")
    train_ds, val_ds, test_ds, target_names = load_speech_commands_dataset(config)
    num_classes = len(target_names)
    
    # Batch datasets
    batch_size = 64
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Determine input shape from first batch
    for x, y in train_ds.take(1):
        input_shape = x.shape[1:]
        break

    print(f"Model Input Shape: {input_shape}")
    print(f"Num Classes: {num_classes}")

    # 2. Build Model
    model = get_tiny_cnn(input_shape, num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.summary()

    # 3. Callbacks
    os.makedirs('../models', exist_ok=True)
    model_save_path = '../models/keyword_cnn.h5'
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=3, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # 4. Train Model
    print(f"\nTraining Model for max {epochs} epochs...")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"\nTraining Complete. Best model saved to {model_save_path}")

if __name__ == '__main__':
    # Fix for memory growth if using GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='Max training epochs')
    args = parser.parse_args()
    main(epochs=args.epochs)
