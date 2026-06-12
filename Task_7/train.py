import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

# Load data
df = pd.read_csv("data.csv")

# Remove unwanted column
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# 🔥 CLEAN TARGET COLUMN
df["Churn"] = df["Churn"].str.strip()

# Drop ALL missing values (after cleaning)
df.dropna(inplace=True)

# Select features
features = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "InternetService",
    "MonthlyCharges", "TotalCharges"
]

df = df[features + ["Churn"]]

# Encode categorical
le_dict = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# Save encoders
pickle.dump(le_dict, open("encoders.pkl", "wb"))

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 🚨 FINAL SAFETY CHECK
print("NaN in y:", y.isnull().sum())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

pickle.dump(scaler, open("scaler.pkl", "wb"))

# Model
model = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=400)
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("✅ Training Complete!")