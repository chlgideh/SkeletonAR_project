import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.model_selection import KFold

def load_ntu_data(data_dir, num_frames=50, num_joints=25):
    x_data, y_data = [], []
    action_labels = []
    
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            file_path = os.path.join(data_dir, file)
            data = np.load(file_path, allow_pickle=True).item()  # (num_frames, num_joints, 3)
            
            if len(data['skel_body0']) < num_frames:
                continue  # Skip short sequences

            # Append the 3D skeleton data of the first body (body0)
            x_data.append(data['skel_body0'][:num_frames].reshape(num_frames, num_joints * 3))
            
            # Extract action label from the data file's directory structure (assuming folder structure as Action/ID)
            label = int(file.split('A')[-1][:3]) - 1  # Example: Extracting action ID from the file name, e.g., "A001"
            y_data.append(label)
            action_labels.append(label)  # Collecting labels for action recognition
    
    return np.array(x_data), np.array(y_data), action_labels

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)
    
    def positional_encoding(self, length, depth):
        depth = depth // 2
        positions = tf.range(length, dtype = tf.float32)[:, tf.newaxis]
        depths = tf.cast(tf.range(depth), tf.float32)[tf.newaxis, :] / tf.cast(depth, tf.float32)
        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates
        pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, x):
        return x + self.pos_encoding[tf.newaxis, :tf.shape(x)[1], :]

class SkeletonTransformer(keras.Model):
    def __init__(self, num_joints=25, num_classes=60, d_model=256, num_layers=4, num_heads=8, dim_feedforward=512, dropout=0.1, sequence_length=50):
        super().__init__()
        self.input_dim = num_joints * 3
        self.embedding = layers.Dense(d_model)
        self.positional_encoding = PositionalEncoding(sequence_length, d_model)
        self.encoder_layers = [
            layers.MultiHeadAttention(num_heads, d_model // num_heads) for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(dropout)
        self.norm_layers = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.fc = layers.Dense(num_classes, activation="softmax")
    
    def call(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for norm, attn in zip(self.norm_layers, self.encoder_layers):
            attn_output = attn(x, x)
            x = norm(x + attn_output)
            x = self.dropout(x)
        x = tf.reduce_mean(x, axis=1)
        return self.fc(x)

def train_with_kfold(model, x_data, y_data, k=5, epochs=10, batch_size=32):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, val_idx in kfold.split(x_data):
        x_train, x_val = x_data[train_idx], x_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]
        
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    
    model.save("skeleton_transformer_all.h5")
    model.save('skeleton_transformer_all.keras')

data_dir = "/mnt/d/ntu_rgb_skeleton_data/nturgb+d_npy/"
x_data, y_data, action_labels = load_ntu_data(data_dir)
if len(x_data) == 0:
    print(f"Error: No data found in {data_dir}.")
else:
    model = SkeletonTransformer(num_joints=25, sequence_length=50, num_classes=60)
    train_with_kfold(model, x_data, y_data)
