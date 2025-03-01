import pandas as pd

# Load the data
df = pd.read_csv('RESULTS/clustered_patients.csv', low_memory=False)

# Basic statistics
num_patients = len(df)
num_clusters = df['cluster'].nunique()

print(f"Total number of patients: {num_patients}")
print(f"Total number of clusters: {num_clusters}\n")

# Distribution of patients across clusters
cluster_counts = df['cluster'].value_counts().sort_index()
print("Number of patients per cluster:")
print(cluster_counts)

# Show top 5 most frequent Diagnóstico Principal per cluster
print("\nTop 5 Diagnóstico Principal per cluster:")
for cluster in sorted(df['cluster'].unique()):
    cluster_df = df[df['cluster'] == cluster]
    top_codes = cluster_df['Diagnóstico Principal'].value_counts().head(5)
    print(f"\nCluster {cluster}:")
    print(top_codes)

# Additional insights: average hospital stay per cluster
df['Estancia Días'] = pd.to_numeric(df['Estancia Días'], errors='coerce')
avg_stay = df.groupby('cluster')['Estancia Días'].mean()

print("\nAverage hospital stay (days) per cluster:")
print(avg_stay.round(2))

