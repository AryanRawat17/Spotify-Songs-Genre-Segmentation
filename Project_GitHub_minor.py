from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'your_dataset.csv' with your file path)
data = pd.read_csv('dataset.csv')



# Check for duplicate track IDs
duplicate_track_ids = data[data.duplicated('track_id')]

# Display duplicate track IDs, if any
if not duplicate_track_ids.empty:
    print("Duplicate Track IDs:")
    print(duplicate_track_ids[['track_id', 'track_name']])
else:
    print("No duplicate track IDs found.")

# Remove duplicates and keep only the first occurrence
data = data.drop_duplicates('track_id', keep='first')




# Explore the statistical distribution of numerical features using box plots
plt.figure(figsize=(16, 8))
plt.subplot(3, 4, 1)
sns.boxplot(x='danceability', data=data)
plt.title('Distribution of Danceability')

plt.subplot(3, 4, 2)
sns.boxplot(x='energy', data=data)
plt.title('Distribution of Energy')

plt.subplot(3, 4, 3)
sns.boxplot(x='key', data=data)
plt.title('Distribution of Key')

plt.subplot(3, 4, 4)
sns.boxplot(x='loudness', data=data)
plt.title('Distribution of Loudness')

plt.subplot(3, 4, 5)
sns.boxplot(x='mode', data=data)
plt.title('Distribution of Mode')

plt.subplot(3, 4, 6)
sns.boxplot(x='speechiness', data=data)
plt.title('Distribution of Speechiness')

plt.subplot(3, 4, 7)
sns.boxplot(x='acousticness', data=data)
plt.title('Distribution of Acousticness')

plt.subplot(3, 4, 8)
sns.boxplot(x='instrumentalness', data=data)
plt.title('Distribution of Instrumentalness')

plt.subplot(3, 4, 9)
sns.boxplot(x='liveness', data=data)
plt.title('Distribution of Liveness')

plt.subplot(3, 4, 10)
sns.boxplot(x='valence', data=data)
plt.title('Distribution of Valence')

plt.subplot(3, 4, 11)
sns.boxplot(x='tempo', data=data)
plt.title('Distribution of Tempo')

plt.subplot(3, 4, 12)
sns.boxplot(x='duration_ms', data=data)
plt.title('Distribution of Duration (ms)')

plt.tight_layout()
plt.show()




# Correlation matrix for numerical features
numerical_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                     'duration_ms', 'track_popularity']
correlation_matrix = data[numerical_columns].corr()

# Visualize correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Relationship between track popularity and audio features
plt.figure(figsize=(14, 6))
sns.pairplot(data, x_vars=numerical_columns[:-1], y_vars=['track_popularity'], kind='scatter')
plt.suptitle('Relationship Between Track Popularity and Audio Features')
plt.show()

# Patterns in playlist popularity based on genre or subgenre
plt.figure(figsize=(12, 6))
sns.boxplot(x='playlist_genre', y='track_popularity', data=data)
plt.title('Playlist Genre vs. Track Popularity')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='playlist_subgenre', y='track_popularity', data=data)
plt.title('Playlist Subgenre vs. Track Popularity')
plt.xticks(rotation=90)
plt.show()



# For simplicity, we will use StandardScaler from scikit-learn
features_to_scale = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Display the DataFrame with scaled numerical features
print("\nDataFrame with Scaled Numerical Features:")
print(data[features_to_scale].head())


# Select the features for clustering
selected_features = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]

# Scale the features (important for K-Means)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_features)

# Choose the number of clusters (K)
k = 3

# Initialize the KMeans model
kmeans = KMeans(n_clusters=k, init ='k-means++', random_state=0)

# Fit the model to the scaled features
kmeans.fit(scaled_features)

# Add the cluster labels to the original DataFrame
data['cluster_label'] = kmeans.labels_

# Display the DataFrame with cluster labels
print("DataFrame with Cluster Labels:")
print(data.head())

# Visualize the clusters (for the first two features: danceability and energy)
plt.figure(figsize=(10, 6))
plt.scatter(data['danceability'], data['energy'], c=data['cluster_label'], cmap='viridis', alpha=0.5)
plt.title('K-Means Clustering')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.show()

