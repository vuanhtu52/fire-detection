import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

from config import new_size

def get_model_path():
    parser = argparse.ArgumentParser(description="Accept command line arguments")
    
    # Adding arguments
    parser.add_argument("model_path", type=str, help="Path to the model to evaluate")

    # Parse arguments
    args = parser.parse_args()

    return args.model_path

def get_config(model_path):
    # config = {}

    with open(f"{model_path}/config.json") as f:
        config = json.load(f)
    
    # for line in lines:
    #     key = line.split(":")[0].strip()
    #     value = line.split(":")[1].strip()
    #     config[key] = value

    return config


def evaluate(model_path):
    # Get the config of the model
    config = get_config(model_path=model_path)

    # Get the values needed to evaluate the model
    image_size = (new_size.get('width'), new_size.get('height'))
    batch_size = int(config["batch_size"])

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Test", seed=1337, image_size=image_size, batch_size=batch_size, shuffle=True
    )

    # Load the model with the best checkpoint
    best_model = load_model(get_best_checkpoint(model_path=model_path))

    # Calculate loss and accuracy on test set
    # results_eval = best_model.evaluate(test_ds, batch_size=batch_size)
    # for name, value in zip(best_model.metrics_names, results_eval):
    #     print(name, ': ', value)
    # print()

    # Get the class names from the test dataset
    class_names = test_ds.class_names

    # Get the ground truth labels
    y_true = []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
    
    # Get the predicted labels
    y_pred = best_model.predict(test_ds)
    # print("y_pred: ", len(y_pred))
    # print(y_pred)
    # print()
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    # print(y_pred)

    # print("y_pred: ", len(y_pred))
    # print("y_true", len(y_true))

    # Print classification report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)


def get_best_checkpoint(model_path):
    with open(f"{model_path}/logs.txt") as f:
        min_val_loss = 1000000
        best_epoch = None

        current_epoch = None
        for line in f:
            line = line.strip()
            if line.startswith("Epoch"):
                current_epoch = int(line.split(" ")[1][:-1])
            if line.startswith("val_loss"):
                current_val_loss = float(line.split(" ")[1])
                if current_val_loss < min_val_loss:
                    best_epoch = current_epoch
                    min_val_loss = current_val_loss

    print("Best epoch: ", best_epoch)
    print("Min val_loss: ", min_val_loss)

    return f"{model_path}/checkpoints/save_at_{best_epoch}.h5"


def main():
    # Get model path from command line argument
    model_path = get_model_path()
    
    # Evaluate the model
    evaluate(model_path=model_path)

if __name__ == "__main__":
    main()
