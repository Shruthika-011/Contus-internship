import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
df = pd.read_csv("loan_data.csv")
print(df.shape)
print(df.head())
df.fillna(df.median(numeric_only=True), inplace=True)
le = LabelEncoder()
cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

for c in cols:
    df[c] = le.fit_transform(df[c])

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def evaluate(name, y_test, y_pred):
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
evaluate("Logistic Regression", y_test, y_pred_lr)

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
evaluate("Decision Tree", y_test, y_pred_dt)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate("Random Forest", y_test, y_pred_rf)

X_unsup = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_unsup)
print("\nKMeans Clusters:\n", np.bincount(clusters_kmeans))

dbscan = DBSCAN(eps=1.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_unsup)
print("\nDBSCAN Clusters:\n", np.unique(clusters_dbscan, return_counts=True))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_unsup)
print("\nPCA Output:\n", X_pca[:10])