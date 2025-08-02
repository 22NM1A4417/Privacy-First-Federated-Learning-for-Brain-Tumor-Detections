import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall

# Input shape
input_shape = (299, 299, 3)

# Load base Xception model
base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')
base_model.trainable = False  # Freeze base model

# Build custom model on top
inputs = layers.Input(shape=input_shape)
x = base_model(inputs)
x = layers.Flatten()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs, outputs)

# Compile the model (optional here, but useful for preview or testing)
model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

# Save model to file
model.save("xception_transfer_model.h5")
print("✅ Initialized model saved as 'xception_transfer_model.h5'")
