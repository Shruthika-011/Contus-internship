
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("customer_segmentation.csv", encoding='latin-1')
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
print("\nAfter Cleaning Shape:", df.shape)
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

print("\nSample with TotalAmount:\n", df.head())
reference_date = df['InvoiceDate'].max()

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  
    'InvoiceNo': 'count',                                    
    'TotalAmount': 'sum'                                     
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

print("\nRFM Table:\n", rfm.head())
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

print("\nScaled RFM:\n", rfm_scaled[:5])
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

print("\nWCSS Values:\n", wcss)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print("\nCluster Counts:\n", rfm['Cluster'].value_counts())
cluster_summary = rfm.groupby('Cluster').mean()

print("\nCluster Summary:\n", cluster_summary)
plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'])
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Customer Segments")
plt.show()

print("\nFinal RFM with Clusters:\n")
print(rfm.head(10))