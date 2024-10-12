# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:34:15 2024

@author: Jin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:52:21 2024

@author: Jin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from itertools import product
import os
import random
from keras import backend as K
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = -1 + 2 * (data - min_val) / (max_val - min_val)
    return normalized_data

def denormalize(normalized_data, original_min, original_max):
    denormalized_data = (normalized_data + 1) * (original_max - original_min) / 2 + original_min
    return denormalized_data

def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return (1 - SS_res/(SS_tot + K.epsilon()))


WORKING_PATH = os.getcwd()
model_path = os.path.join(WORKING_PATH, 'pre_model.h5')
fine_model_path = os.path.join(WORKING_PATH, 'fine_model.h5')
# fine_model= load_model(fine_model_path, custom_objects={'R2': R2}, by_name=True)

excel_file_path = os.path.join(WORKING_PATH, 'data_test.csv')
data = pd.read_csv(excel_file_path, header=None)

x_values = data.iloc[:, 0].values
x = x_values[:100,]
z_values = data.iloc[:, 1].values
z = pd.Series(z_values)
z.drop_duplicates(inplace=True)
z = z.tolist()
z = np.array(z)
t = np.arange(0, 50.5, 0.5)  

# plt.scatter(x_values, z_values)


points = list(product(x, z))

input_data = []
for point in points:
    x_val, z_val = point
    t_mesh, x_mesh, z_mesh = np.meshgrid(t, x_val, z_val, indexing='ij')
    input_data.append(np.column_stack((t_mesh.flatten(), x_mesh.flatten(), z_mesh.flatten(),)))

input_data = np.concatenate(input_data)
input_data[:, [0, 2]] = input_data[:, [2, 0]]
output_data = data.iloc[:, 2:].values.flatten()
output_data = np.maximum(output_data, 0)
output_data = np.minimum(output_data, 1)
T = output_data.reshape((100, 100, 101))

# number of selected points
num_selected_points = 10

# Select data points randomly
indices = random.sample(range(len(points)), num_selected_points)

selected_points = np.array(points)[indices]

# # Visualize the selected points
plt.figure(figsize=(4, 3))
plt.scatter(selected_points[:, 1], selected_points[:, 0], facecolors='none', edgecolors='r', marker='o', s=30, linewidths=1)
plt.title('Observation points',  fontdict={'fontsize': 10, 'fontweight': 'bold', 'fontfamily': 'Times new roman'})
plt.xlabel('X(m)',  fontdict={'fontsize': 8, 'fontweight': 'bold', 'fontfamily': 'Times new roman'})
plt.ylabel('Z(m)',  fontdict={'fontsize': 8, 'fontweight': 'bold', 'fontfamily': 'Times new roman'})
plt.xticks(fontsize=6, fontfamily='Times new roman', fontweight='bold')  
plt.yticks(fontsize=6, fontfamily='Times new roman', fontweight='bold')  
plt.xlim([0, 10])  
plt.ylim([0, 10])  
ax = plt.gca()
ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
plt.savefig(WORKING_PATH+'/Observation points_PITL.jpg',format='JPG', dpi=600)
plt.show()

selected_x = []
selected_z = []

for _ in range(num_selected_points):
    x_idx = random.randint(0, len(x) - 1)
    z_idx = random.randint(0, len(z) - 1)
    selected_x.append(x[x_idx])
    selected_z.append(z[z_idx])


selected_temperature_sequences = []
selected_time_sequences = []

for i in range(num_selected_points):
    x_val = selected_x[i]
    z_val = selected_z[i]

    x_idx = np.where(x == x_val)[0][0]
    z_idx = np.where(z == z_val)[0][0]

    selected_time_sequences.append(t)
    selected_temperature_sequences.append(T[x_idx, z_idx, :])
    
num_time_steps = 101
selected_time_sequences = np.array(selected_time_sequences)
selected_temperature_sequences = np.array(selected_temperature_sequences)
repeated_x = np.repeat(selected_x, num_time_steps)
repeated_z = np.repeat(selected_z, num_time_steps)
repeated_t = np.tile(t, num_selected_points)


input_data_select = np.column_stack((repeated_x, repeated_z, repeated_t))
output_data_select = selected_temperature_sequences.flatten()
output_data_select = np.maximum(output_data_select, 0)

# Split the data into training and testing sets
X_train_fine, X_test_fine, y_train_fine, y_test_fine = train_test_split(input_data_select, output_data_select,
                                                                        test_size=0.2, random_state=42)

# Define the model architecture
input_shape = (3,)  # You may need to adjust this based on your input shape
pre_model = models.Sequential([
    layers.Dense(12, activation='relu', input_shape = input_shape),
    # layers.Dropout(0.1),
    layers.Dense(12, activation='relu'),
    # layers.Dropout(0.1),
    layers.Dense(12, activation='relu'),
    # layers.Dropout(0.1),
    layers.Dense(12, activation='relu'),
    # layers.Dropout(0.1),
    layers.Dense(12, activation='relu'),
    # layers.Dropout(0.1),
    layers.Dense(12, activation='relu'),
    # layers.Dropout(0.1),
    layers.Dense(1, activation='linear')  
])

# Load pre-trained weights
pre_model.load_weights(model_path)

for layer in pre_model.layers:
    layer.trainable = False

# pre_model.layers[5].trainable = True   
# pre_model.layers[4].trainable = True 
# pre_model.layers[2].trainable = True
# pre_model.layers[1].trainable = True    
pre_model.layers[0].trainable = True    


# # Compile the model
LEARNING_RATE = 0.001
EPOCH_NUMBER = 1500


es     = callbacks.EarlyStopping(monitor='val_R2', mode='max', verbose=1, patience=50, 
                                  min_delta=0.01, restore_best_weights=True)
reduce = callbacks.ReduceLROnPlateau(monitor='val_R2', factor=0.5, patience=50, verbose=1, 
                                      mode='max', min_delta=0.01, cooldown=0, min_lr=LEARNING_RATE / 10)
tnan   = callbacks.TerminateOnNaN()

pre_model.compile(loss='mse', metrics=[R2], optimizer=optimizers.Adam(learning_rate=LEARNING_RATE))

history_callback = pre_model.fit(X_train_fine, y_train_fine, epochs=EPOCH_NUMBER, batch_size=512, 
                                      validation_split=0.2, verbose=2,)

pre_model.save_weights(fine_model_path, overwrite=True)
history = history_callback.history

# Plot training loss
plt.figure(figsize=(4, 3))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


predicted_T = pre_model.predict(input_data).flatten()
predicted_T = predicted_T.reshape((100, 100, 101))
real_T = output_data.reshape((100, 100, 101))


selected_x_idx = 50
selected_z_idx = 50

selected_x_point = x[selected_x_idx]
selected_z_point = z[selected_z_idx]

x_idx_point = np.where(x == selected_x_point)[0][0]
z_idx_point = np.where(z == selected_z_point)[0][0]

actual_temperature_sequence = real_T[x_idx_point, z_idx_point, :]
predicted_temperature_sequence = predicted_T[x_idx_point, z_idx_point, :]

plt.figure(figsize=(4, 3))
plt.plot(t, actual_temperature_sequence, label='Actual Temperature')
plt.plot(t, predicted_temperature_sequence, label='Predicted Temperature', linestyle='dashed')
plt.title(f'Temperature Sequence at Point ({selected_x_point}, {selected_z_point})')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()

import matplotlib.font_manager as fm

rmse_matrix = np.zeros((100, 100, 101))

for i in range(len(x)):
    for j in range(len(z)):
        for k in range(len(t)):
            actual_temperature = real_T[i, j, k]
            predicted_temperature = predicted_T[i, j, k]
 
            actual_temperature = np.array(actual_temperature).flatten()
            predicted_temperature = np.array(predicted_temperature).flatten()

            rmse_matrix[i, j, k] = np.sqrt(mean_squared_error(actual_temperature, predicted_temperature))
            
average_rmse = np.mean(rmse_matrix, axis=2)


#  RMSE 
plt.figure(figsize=(4, 3))
image = plt.imshow(rmse_matrix[:,:,40], cmap='seismic', extent=[0, 10, 0, 10], vmin=0, vmax=1.0, aspect='auto', origin='lower', interpolation='bilinear')
colorbar = plt.colorbar(image, label='RMSE')
font_prop = fm.FontProperties(family='Times new roman', style='normal', weight='bold')
for label in colorbar.ax.get_yticklabels():
    label.set_fontproperties(font_prop)
colorbar.set_label('RMSE', fontproperties=font_prop, fontsize=8, fontweight='bold')
colorbar.ax.tick_params(labelsize=6, width=1)
# colorbar.set_label('RMSE', fontsize=8, fontweight='bold', fontfamily='Times new roman')
plt.xlabel('X(m)',  fontdict={'fontsize': 8, 'fontweight': 'bold', 'fontfamily': 'Times new roman'})
plt.ylabel('Z(m)',  fontdict={'fontsize': 8, 'fontweight': 'bold', 'fontfamily': 'Times new roman'})
plt.xticks(fontsize=6, fontfamily='Times new roman', fontweight='bold')  
plt.yticks(fontsize=6, fontfamily='Times new roman', fontweight='bold')  
# ax = plt.gca()
ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
plt.savefig(WORKING_PATH+'/RMSE_PITL_1.jpg',format='JPG', dpi=600)
plt.show()


selected_rmse = np.zeros(len(selected_x))

for i, x_val in enumerate(selected_x):
    z_val = selected_z[i]
    x_idx = np.where(x == x_val)[0][0]
    z_idx = np.where(z == z_val)[0][0]

    actual_temperature_sequence = real_T[x_idx, z_idx, :]
    predicted_temperature_sequence = predicted_T[x_idx, z_idx, :]

    rmse = np.sqrt(mean_squared_error(actual_temperature_sequence, predicted_temperature_sequence))
    selected_rmse[i] = rmse

# scatter
plt.figure(figsize=(8, 6))
plt.scatter(selected_x, selected_z, c=selected_rmse, cmap='seismic', marker='o', s=30, linewidths=1)
plt.colorbar(label='RMSE')
plt.xlabel('X(m)')
plt.ylabel('Z(m)')
plt.title('Spatial Distribution of RMSE')
plt.show()

