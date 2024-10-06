from google.colab import files

# Upload the catalog file
print("Please upload your seismic catalog file:")
uploaded = files.upload()

# Upload the seismic waveform files
print("Please upload your seismic waveform CSV files:")
uploaded_seismic = files.upload()


import pandas as pd

# Assuming the uploaded catalog file is named 'apollo12_catalog_GradeA_final.csv'
catalog_file_path = 'apollo12_catalog_GradeA_final.csv'

# Load the catalog file
catalog = pd.read_csv(catalog_file_path)

# Preview the first few rows of the catalog
catalog.head()


import os

# Function to load seismic data files (CSV format)
def load_seismic_data(filename):
    csv_file = f"{filename}.csv"
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        return data
    else:
        print(f"File {csv_file} not found.")
        return None

# Example: Load data for the first event
first_event = catalog.iloc[0]
seismic_data = load_seismic_data(first_event['filename'])

# Preview the loaded seismic data
seismic_data.head()




from datetime import datetime

# Convert 'time_abs' to datetime format
catalog['time_abs'] = pd.to_datetime(catalog['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], format='%Y-%m-%dT%H:%M:%S.%f')

# Extract additional time-based features
catalog['day_of_week'] = catalog['time_abs'].dt.dayofweek
catalog['hour_of_day'] = catalog['time_abs'].dt.hour

# Preview the modified catalog
catalog.head()


import numpy as np
from obspy.signal.trigger import classic_sta_lta

# Function to extract features from seismic waveform data
def extract_features(seismic_df):
    features = {}

    # Basic statistics
    features['mean_velocity'] = seismic_df['velocity(m/s)'].mean()
    features['std_velocity'] = seismic_df['velocity(m/s)'].std()
    features['max_velocity'] = seismic_df['velocity(m/s)'].max()
    features['min_velocity'] = seismic_df['velocity(m/s)'].min()

    # Frequency domain features using FFT
    fft_vals = np.fft.fft(seismic_df['velocity(m/s)'])
    fft_freq = np.fft.fftfreq(len(fft_vals), d=(seismic_df['time_rel(sec)'][1] - seismic_df['time_rel(sec)'][0]))

    pos_mask = fft_freq > 0
    fft_freq = fft_freq[pos_mask]
    fft_power = np.abs(fft_vals[pos_mask])

    features['spectral_centroid'] = np.sum(fft_freq * fft_power) / np.sum(fft_power)
    features['spectral_bandwidth'] = np.sqrt(np.sum(((fft_freq - features['spectral_centroid'])**2) * fft_power) / np.sum(fft_power))

    # STA/LTA feature
    df = 1 / ((seismic_df['time_rel(sec)'][1] - seismic_df['time_rel(sec)'][0]))
    sta_len = 120
    lta_len = 600
    cft = classic_sta_lta(seismic_df['velocity(m/s)'], int(sta_len * df), int(lta_len * df))
    features['sta_lta_max'] = np.max(cft)
    features['sta_lta_mean'] = np.mean(cft)

    return features

# Example: Extract features for the first seismic event
features_first = extract_features(seismic_data)
features_first


# Initialize a list to store feature dictionaries
feature_list = []

# Iterate through each event in the catalog and extract features
for index, row in catalog.iterrows():
    filename = row['filename']
    seismic_df = load_seismic_data(filename)
    if seismic_df is not None:
        features = extract_features(seismic_df)
        features['time_rel(sec)'] = row['time_rel(sec)']
        features['day_of_week'] = row['day_of_week']
        features['hour_of_day'] = row['hour_of_day']
        features['mq_type'] = row['mq_type']
        feature_list.append(features)

# Convert the list of feature dictionaries to a DataFrame
features_df = pd.DataFrame(feature_list)

# Preview the extracted features
features_df.head()


from sklearn.preprocessing import LabelEncoder

# Encode the target labels (mq_type) as numerical values
le = LabelEncoder()
features_df['mq_type_encoded'] = le.fit_transform(features_df['mq_type'])

# Define the feature columns and target column
feature_columns = ['mean_velocity', 'std_velocity', 'max_velocity', 'min_velocity',
                   'spectral_centroid', 'spectral_bandwidth', 'sta_lta_max', 'sta_lta_mean',
                   'time_rel(sec)', 'day_of_week', 'hour_of_day']

X = features_df[feature_columns]
y = features_df['mq_type_encoded']

# Preview the feature matrix and target vector
X.head(), y.head()



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display the classification report
print(classification_report(y_test, y_pred, target_names=le.classes_))


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
