import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score


def evaluate_multimodal_model(model, dataset, class_names, output_path="evaluation_report.txt"):
    """
    Evaluate a multimodal model on a given dataset and save the results to a file.

    Parameters:
        model (tf.keras.Model): Trained multimodal model to evaluate.
        dataset (tf.data.Dataset): Dataset in the format ((spec, face), label).
        class_names (List[str]): List of class names for classification report.
        output_path (str): Path to the output text file to save results.
    """
    y_true, y_pred = [], []

    for (spec_batch, face_batch), label_batch in dataset:
        preds = model.predict([spec_batch, face_batch], verbose=0)
        y_true.extend(label_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)

    # Print to console
    print(f'\nBalanced Accuracy:\t{balanced_acc:.4f}')
    print("\nClassification Report:")
    print(report)

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Balanced Accuracy:\t{balanced_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\nEvaluation results saved to: {output_path}")
