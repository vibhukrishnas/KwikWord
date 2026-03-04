import os
import tensorflow as tf
import numpy as np
from dataset import load_speech_commands_dataset, Config
from sklearn.metrics import confusion_matrix

def main():
    config = Config()
    
    model_path = '../models/keyword_cnn.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train.py first.")
        return

    print("Loading test dataset...")
    _, _, test_ds, target_names = load_speech_commands_dataset(config)
    
    batch_size = 64
    test_ds_batched = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"Loading best model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    model.summary()
    
    print("\n--- Evaluation ---")
    loss, accuracy = model.evaluate(test_ds_batched, verbose=1)
    
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    print("\n--- Confusion Matrix ---")
    # Collect all true labels and predictions to compute the matrix
    all_labels = []
    all_preds = []
    
    for xs, ys in test_ds_batched:
        preds = model.predict(xs, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        
        all_labels.extend(ys.numpy())
        all_preds.extend(pred_labels)
        
    cm = confusion_matrix(all_labels, all_preds)
    
    print("Class mapping:")
    for i, name in enumerate(target_names):
        print(f"{i}: {name}")
        
    print("\nConfusion Matrix array:")
    print(cm)

if __name__ == '__main__':
    main()
