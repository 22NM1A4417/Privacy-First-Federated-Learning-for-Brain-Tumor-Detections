import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Path to your test dataset - update accordingly
test_data_dir = r'D:\Brain_Tumor\Brain-Tumor-Classification-using-Federated-Learning-Diffusion-Models\data\final_test_data'

# Load saved model
model = tf.keras.models.load_model('global_model_round5.h5')

# Preprocessing for test data (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for correct label order
)

# Evaluate model on test data
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Predictions for confusion matrix and classification report
test_predictions = model.predict(test_generator)
predicted_classes = np.argmax(test_predictions, axis=1)
true_classes = test_generator.classes

cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(cm)

report = classification_report(true_classes, predicted_classes, target_names=list(test_generator.class_indices.keys()))
print("Classification Report:")
print(report)
