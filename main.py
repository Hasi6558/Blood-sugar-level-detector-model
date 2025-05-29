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

# Update paths for local execution
base_dir = r'c:\Users\Hasindu Shehan\Desktop\Research ML project'
train_data_path = os.path.join(base_dir, 'train_data.csv')
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

model.summary()  # Show model architecture summary
print(f"Number of layers in the model: {len(model.layers)}")

# exit()  # Stop execution after showing the summary

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

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

# --- Keras Tuner for Hyperparameter Tuning ---
# Uncomment the following block to use Keras Tuner
'''
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(
        filters=hp.Int('filters', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('kernel_size', values=[3, 5]),
        activation='relu',
        input_shape=(image_size[0], image_size[1], 3)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mse',
        metrics=['mae']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory=base_dir,
    project_name='keras_tuner_sugar_level'
)

tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

best_model = tuner.get_best_models(num_models=1)[0]
# You can now evaluate or train best_model further
'''
