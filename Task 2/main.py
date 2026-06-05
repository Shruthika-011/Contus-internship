from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Access database & collection
db = client["Movie_review"]
collection = db["movies"]

# Fetch data
movies = list(collection.find())

if len(movies) == 0:
    print("No data found")
else:
    df = pd.DataFrame(movies)

    # Convert ObjectId to string
    df["_id"] = df["_id"].astype(str)

    # Handle genre (array → string)
    df["genre"] = df["genre"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )

    # Extract review count
    df["review_count"] = df["reviews"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    print(df[["title", "genre", "rating", "review_count"]])