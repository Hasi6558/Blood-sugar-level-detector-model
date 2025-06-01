import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Update paths for local execution
base_dir = '/home/hasindu-shehan/Desktop/Blood sugar level detector/Blood-sugar-level-detector-model'
train_data_path = 'train_data.csv'
data_dir = os.path.join(base_dir, 'Data')

train_data = pd.read_csv(train_data_path)

image_size = (64, 64)

images = []
sugar_levels = []

for _, row in train_data.iterrows():
    folder = row['Folder']
    sugar_level = row['Sugar_Level']
    folder_path = os.path.join(data_dir, folder)

    for file_name in os.listdir(folder_path):
        # Only process image files
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            continue
        file_path = os.path.join(folder_path, file_name)
        try:
            # Load and preprocess the image
            img = load_img(file_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array)
            sugar_levels.append(sugar_level)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")

X = np.array(images)
y = np.array(sugar_levels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.summary()  # Show model architecture summary
print(f"Number of layers in the model: {len(model.layers)}")

# exit()  # Stop execution after showing the summary

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Add early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Update model training to include early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

y_pred = model.predict(X_test).flatten()

# Calculate and print R^2 value
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2:.4f}")

# 1. Loss Curves (Training vs Validation)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'loss_curve.png'))
plt.close()

# 2. Mean Absolute Error Curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training vs Validation MAE')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'mae_curve.png'))
plt.close()

# 3. Predicted vs Actual Values
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Sugar Level')
plt.ylabel('Predicted Sugar Level')
plt.title('Predicted vs Actual Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'pred_vs_actual.png'))
plt.close()

# 4. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sugar Level')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'residual_plot.png'))
plt.close()

# 5. Prediction Error Line Plot
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test, label='Actual', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Predicted', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Sugar Level')
plt.title('Prediction Error Line Plot')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'prediction_error_line.png'))
plt.close()

# 6. Distribution of Actual vs Predicted Values
plt.figure(figsize=(8, 5))
sns.kdeplot(y_test, label='Actual', fill=True)
sns.kdeplot(y_pred, label='Predicted', fill=True)
plt.xlabel('Sugar Level')
plt.ylabel('Density')
plt.title('Distribution of Actual vs Predicted Values')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'distribution_plot.png'))
plt.close()

model_path = os.path.join(base_dir, 'sugar_level_model.keras')
model.save(model_path)

print(f"Model saved to {model_path}")

# Commenting out hyperparameter tuning and using the baseline model as the main model
# --- Keras Tuner for Hyperparameter Tuning ---
# Define the model-building function for Keras Tuner
# def build_model(hp):
#     model = Sequential()
#     model.add(tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)))
#     # Tune the number of Conv2D layers
#     for i in range(hp.Int('num_conv_layers', min_value=1, max_value=3)):
#         model.add(Conv2D(
#             filters=hp.Int(f'filters_{i}', min_value=16, max_value=64, step=16),
#             kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
#             activation='relu'
#         ))
#         model.add(MaxPooling2D(pool_size=(2, 2)))

#     # Tune the dropout rate
#     model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
#     model.add(Flatten())

#     # Tune the number of dense units
#     model.add(Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
#     model.add(Dense(1))  # Output layer for regression

#     # Tune the optimizer and learning rate
#     optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
#     learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#     if optimizer == 'adam':
#         opt = Adam(learning_rate=learning_rate)
#     elif optimizer == 'sgd':
#         opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
#     elif optimizer == 'rmsprop':
#         opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

#     model.compile(optimizer=opt, loss='mse', metrics=['mae'])
#     return model

# Initialize the Keras Tuner
# tuner = kt.RandomSearch(
#     build_model,
#     objective='val_loss',  # Optimize for validation loss
#     max_trials=10,  # Number of different hyperparameter combinations to try
#     executions_per_trial=1,  # Number of models to train per trial
#     directory=base_dir,  # Directory to save tuner results
#     project_name='keras_tuner_sugar_level'
# )

# Perform the search
# tuner.search(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32, verbose=1)

# Retrieve the best hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print(f"""
# The optimal number of Conv2D layers is {best_hps.get('num_conv_layers')},
# the optimal number of filters per layer is {[best_hps.get(f'filters_{i}') for i in range(best_hps.get('num_conv_layers'))]},
# the optimal kernel sizes are {[best_hps.get(f'kernel_size_{i}') for i in range(best_hps.get('num_conv_layers'))]},
# the optimal dropout rate is {best_hps.get('dropout')},
# the optimal number of dense units is {best_hps.get('dense_units')},
# the optimal optimizer is {best_hps.get('optimizer')},
# the optimal learning rate is {best_hps.get('learning_rate')}.
# """)

# Train the baseline model as the main model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the baseline model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Save the baseline model
baseline_model_path = os.path.join(base_dir, 'baseline_sugar_level_model.keras')
model.save(baseline_model_path)
print(f"Baseline model saved to {baseline_model_path}")
