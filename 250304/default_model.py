import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import os
from sklearn.model_selection import KFold

def filter_files_by_action(data_dir, action_tag):
    files = [f for f in os.listdir(data_dir) if f.endswith(".npy") and action_tag in f]
    return [os.path.join(data_dir, f) for f in files]

def load_ntu_data(files, num_frames=50, num_joints=25):
    x_data, y_data = [], []
    for file in files:
        data = np.load(file, allow_pickle=True).item()  # (num_frames, num_joints, 3)
        if len(data['skel_body0']) < num_frames:
            continue  # Skip short sequences
        # x_data.append(data[:num_frames].reshape(num_frames, num_joints * 3))
        # x_data.append(data['skel_body0'][:num_frames].reshape(num_frames, num_joints * 3)) 
        x_data.append(data['skel_body0'][:num_frames].transpose(1,0,2)) #transpose for (50,25,3) -> (25,50,3)
        y_data.append(int(file.split('A')[-1][:3]) - 1)  # Extract action label
    return np.array(x_data), np.array(y_data)

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
        # return x + self.pos_encoding[tf.newaxis, :tf.shape(x)[1], :]
        return x + self.pos_encoding[tf.newaxis, :, :, :]

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
    
    # model.save("skeleton_transformer.h5")
    # model.save('skeleton_transformer.keras')
    model.save("skeleton_transpose_transformer.h5")
    model.save('skeleton_transpose_transformer.keras')


data_dir = "/mnt/d/ntu_rgb_skeleton_data/nturgb+d_npy/"
action_tag = "A001"
files = filter_files_by_action(data_dir, action_tag)
if not files:
    print(f"Error: No files found for action {action_tag}.")
else:
    x_data, y_data = load_ntu_data(files)
    model = SkeletonTransformer(num_joints=25, sequence_length=50, num_classes=60)
    train_with_kfold(model, x_data, y_data)
