import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall

# Privacy-first: Use Differential Privacy (DP) optimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

class FederatedLearning:
    def __init__(self, dataset_root, test_data_dir):
        self.clients = [f"client_{i}" for i in range(1, 7)]
        self.dataset_root = dataset_root
        self.test_data_dir = test_data_dir
        self.global_model = self.initialize_model()
        self.client_data = self.load_client_data()
        self.test_data = self.load_test_data()

    def initialize_model(self):
        model = load_model("xception_transfer_model.h5")
        model.trainable = True
        for layer in model.layers[:100]:
            layer.trainable = False
        return model

    def normalize_img(self, x, y):
        return tf.cast(x, tf.float32) / 255.0, y

    def get_augmented_dataset(self, dataset):
        data_augmentation = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),
        ])
        return dataset.map(lambda x, y: (data_augmentation(x, training=True), y)).map(self.normalize_img)

    def load_client_data(self):
        client_data = {}
        for client in self.clients:
            client_path = os.path.join(self.dataset_root, client)
            if os.path.exists(client_path):
                dataset = image_dataset_from_directory(
                    client_path,
                    image_size=(299, 299),
                    batch_size=32,
                    label_mode="categorical",
                    shuffle=True
                )
                dataset = self.get_augmented_dataset(dataset)
                client_data[client] = dataset
            else:
                print(f"Warning: {client_path} does not exist.")
        return client_data

    def load_test_data(self):
        if os.path.exists(self.test_data_dir):
            test_dataset = image_dataset_from_directory(
                self.test_data_dir,
                image_size=(299, 299),
                batch_size=32,
                label_mode="categorical",
                shuffle=False
            ).map(self.normalize_img)
            return test_dataset
        else:
            print(f"Warning: Test data directory {self.test_data_dir} does not exist.")
            return None

    def train_client_model(self, client_name):
        local_model = clone_model(self.global_model)
        local_model.set_weights(self.global_model.get_weights())

        # Privacy-First: Using Differential Privacy optimizer (DP-SGD)
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=1.0,
            noise_multiplier=1.1,
            num_microbatches=32,
            learning_rate=0.0005
        )

        local_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

        early_stop = EarlyStopping(monitor='accuracy', patience=2, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f"best_{client_name}.h5", monitor='accuracy', save_best_only=True)

        history = local_model.fit(
            self.client_data[client_name],
            epochs=5,
            callbacks=[early_stop, checkpoint],
            verbose=0
        )

        return local_model.get_weights(), history.history['accuracy'][-1]

    def aggregate_client_updates(self, client_weights):
        new_weights = []
        for weights in zip(*client_weights):
            new_weights.append(np.mean(weights, axis=0))
        self.global_model.set_weights(new_weights)

    def evaluate_global_model(self):
        if self.test_data is None:
            print("No test dataset found, skipping evaluation.")
            return None
        self.global_model.compile(
            optimizer=DPKerasAdamOptimizer(
                l2_norm_clip=1.0,
                noise_multiplier=1.1,
                num_microbatches=32,
                learning_rate=0.0005
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        results = self.global_model.evaluate(self.test_data, verbose=0)
        metrics_names = self.global_model.metrics_names
        eval_results = dict(zip(metrics_names, results))
        print(f"\n📈 Global model evaluation on test data: {eval_results}")
        return eval_results

    def simulate(self, num_rounds=10):
        for round_num in range(num_rounds):
            print(f"\n🔁 Federated Round {round_num + 1}")
            client_weights = []
            client_accuracies = {}

            for client in self.clients:
                if client in self.client_data:
                    print(f"📶 Training on {client}")
                    weights, acc = self.train_client_model(client)
                    client_weights.append(weights)
                    client_accuracies[client] = acc
                else:
                    print(f"⚠️ No data found for {client}, skipping training.")

            if client_weights:
                self.aggregate_client_updates(client_weights)

            print(f"📊 Client training accuracies for round {round_num + 1}:")
            for client, acc in client_accuracies.items():
                print(f"  - {client}: {acc:.4f}")

            self.evaluate_global_model()

        # ✅ Save the final global model
        self.global_model.save("federated_global_model.h5")
        print("\n✅ Federated training complete. Final model saved as 'federated_global_model.h5'.")

if __name__ == "__main__":
    dataset_root = r'D:\Brain_Tumor\Brain-Tumor-Classification-using-Federated-Learning-Diffusion-Models\data\final_clients'
    test_data_dir = r'D:\Brain_Tumor\Brain-Tumor-Classification-using-Federated-Learning-Diffusion-Models\data\final_test_data'
    fl = FederatedLearning(dataset_root, test_data_dir)
    fl.simulate(num_rounds=10)
