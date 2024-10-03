# Section 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.fft import fft

# Set aesthetic style for seaborn
sns.set(style="whitegrid")

# Section 2: Load the Dataset
# Load the dataset (Make sure to upload your CSV file in Colab)
from google.colab import files
uploaded = files.upload()

# Load the dataset into a DataFrame
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)

# Display the first few rows of the dataset
print(data.head())

# Section 3: Data Preparation
# Drop the non-numerical column (Planet/Moon) and define features (X) and target (y)
X = data.drop(columns=['Planet/Moon', 'Seismic Anomaly (0/1)'])
y = data['Seismic Anomaly (0/1)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Section 4: Model Training
# Initialize the XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model
xgb_model.fit(X_train, y_train)

# Section 5: Predictions and Evaluations
# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print accuracy and classification report
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Section 6: Feature Importance Visualization
plt.figure(figsize=(10,6))
xgboost_importances = xgb_model.feature_importances_
sns.barplot(x=xgboost_importances, y=X.columns)
plt.title('Feature Importance in Seismic Anomaly Prediction')
plt.show()

# Section 7: Multiple Plots in a Subplot
fig, axes = plt.subplots(2, 2, figsize=(15,10))

# Plot 1: Scatter Plot of Seismic Magnitude vs Crust Thickness
sns.scatterplot(ax=axes[0,0], x=X_test['Seismic Magnitude'], y=X_test['Crust Thickness (km)'], hue=y_pred, style=y_test)
axes[0,0].set_title('Seismic Magnitude vs Crust Thickness')
axes[0,0].set_xlabel('Seismic Magnitude')
axes[0,0].set_ylabel('Crust Thickness (km)')

# Plot 2: Pretty Plot (Seaborn's aesthetic scatterplot)
sns.lmplot(x='Gravitational Force (N)', y='Depth (km)', hue='Seismic Anomaly (0/1)', data=data)
plt.title('Gravitational Force vs Depth')

# Plot 3: Line Plot - Seismic Magnitude over Depth
axes[1,0].plot(X_test['Depth (km)'], X_test['Seismic Magnitude'], label='Seismic Magnitude over Depth')
axes[1,0].set_title('Seismic Magnitude vs Depth')
axes[1,0].set_xlabel('Depth (km)')
axes[1,0].set_ylabel('Seismic Magnitude')

# Plot 4: Barplot for Volcanic Activity
sns.barplot(ax=axes[1,1], x=X_test['Volcanic Activity (0-2)'], y=X_test['Seismic Magnitude'], hue=y_pred)
axes[1,1].set_title('Volcanic Activity vs Seismic Magnitude')
plt.tight_layout()

# Section 8: Spectrogram Visualization
# Function to generate a mock seismic wave signal
def generate_seismic_wave(frequency=5, samples=1000, sampling_rate=100):
    t = np.linspace(0, 10, samples)
    wave = np.sin(2 * np.pi * frequency * t)  # Sine wave representing seismic activity
    return t, wave

# Generate a mock seismic wave signal
t, wave = generate_seismic_wave()

# Spectrogram
plt.figure(figsize=(10, 6))
plt.specgram(wave, NFFT=256, Fs=100, noverlap=128)
plt.title('Seismic Wave Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Intensity [dB]')
plt.show()

# Section 9: Characteristic Function Plot
# Fourier Transform (FFT) to simulate a characteristic seismic wave function
fft_wave = fft(wave)
frequencies = np.fft.fftfreq(len(wave), d=t[1]-t[0])

plt.figure(figsize=(10,6))
plt.plot(frequencies, np.abs(fft_wave))
plt.title('Seismic Wave Characteristic Function')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()

# Section 10: User Input Functionality
def get_user_input():
    seismic_magnitude = float(input("Enter Seismic Magnitude: "))         # Sample Input: 7.0
    depth = float(input("Enter Depth (km): "))                            # Sample Input: 20.0
    wave_height = float(input("Enter Wave Height (m): "))                # Sample Input: 2.5
    gravitational_force = float(input("Enter Gravitational Force (N): ")) # Sample Input: 9.81
    crust_thickness = float(input("Enter Crust Thickness (km): "))        # Sample Input: 40.0
    volcanic_activity = int(input("Enter Volcanic Activity (0-2): "))     # Sample Input: 1
    tidal_forces = float(input("Enter Tidal Forces (N): "))               # Sample Input: 12.0
    lithosphere_density = float(input("Enter Lithosphere Density (g/cm³): ")) # Sample Input: 3.0
    tectonic_activity = int(input("Enter Tectonic Activity (0/1): "))      # Sample Input: 1
    
    # Return the inputs as a DataFrame
    return pd.DataFrame({
        'Seismic Magnitude': [seismic_magnitude],
        'Depth (km)': [depth],
        'Wave Height (m)': [wave_height],
        'Gravitational Force (N)': [gravitational_force],
        'Crust Thickness (km)': [crust_thickness],
        'Volcanic Activity (0-2)': [volcanic_activity],
        'Tidal Forces (N)': [tidal_forces],
        'Lithosphere Density (g/cm³)': [lithosphere_density],
        'Tectonic Activity (0/1)': [tectonic_activity]
    })

# Get user input and make a prediction
user_input = get_user_input()
prediction = xgb_model.predict(user_input)

# Display the prediction
print(f"Predicted Seismic Anomaly: {'Yes' if prediction[0] == 1 else 'No'}")

# Visualize user input's scatter plot for Seismic Magnitude and Crust Thickness
plt.figure(figsize=(8,6))
sns.scatterplot(x=user_input['Seismic Magnitude'], y=user_input['Crust Thickness (km)'], color='red', label='User Input')
plt.title('Seismic Magnitude vs Crust Thickness - User Input')
plt.xlabel('Seismic Magnitude')
plt.ylabel('Crust Thickness (km)')
plt.legend()
plt.show()
