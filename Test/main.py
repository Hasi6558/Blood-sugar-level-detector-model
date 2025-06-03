import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'Data')
train_data_path = os.path.join(base_dir, '..', 'train_data.csv')

# Parameters
image_size = (64, 64)

# Load train_data.csv to get actual sugar levels
train_data = pd.read_csv(train_data_path)
folder_to_sugar = dict(zip(train_data['Folder'], train_data['Sugar_Level']))

images = []
sample_ids = []
actual_values = []

# Loop through all folders in Data (one image per folder)
for folder in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    # Find the first image file in the folder
    image_file = None
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_file = file_name
            break
    if image_file is None:
        continue
    file_path = os.path.join(folder_path, image_file)
    try:
        img = load_img(file_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        sample_ids.append(folder)
        actual_values.append(folder_to_sugar.get(folder, np.nan))
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")

# Dummy model: predicted = actual + noise (for demonstration)
# In real use, replace this with your model's predictions
np.random.seed(42)
predicted_values = np.array(actual_values) + np.random.normal(0, 10, size=len(actual_values))

# Create DataFrame
results_df = pd.DataFrame({
    'Sample Index': range(len(images)),
    'Sample ID': sample_ids,
    'Actual Value': actual_values,
    'Predicted Value': predicted_values
})

# Save to CSV
csv_path = os.path.join(base_dir, 'predicted_vs_actual.csv')
results_df.to_csv(csv_path, index=False)
print(f"CSV file saved to {csv_path}")

# Plot graph
plt.figure(figsize=(10, 5))
plt.plot(results_df['Sample Index'], results_df['Actual Value'], label='Actual', marker='o')
plt.plot(results_df['Sample Index'], results_df['Predicted Value'], label='Predicted', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Sugar Level')
plt.title('Predicted vs Actual Values by Sample Index')
plt.legend()
plt.tight_layout()
graph_path = os.path.join(base_dir, 'predicted_vs_actual_graph.png')
plt.savefig(graph_path)
plt.close()
print(f"Graph saved to {graph_path}")
