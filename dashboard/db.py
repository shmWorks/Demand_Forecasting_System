from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "retail_iq_dashboard"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]

    # Initialize targeted indexes
    db.users.create_index([("username", ASCENDING)], unique=True)
    db.metrics.create_index([("date", DESCENDING)])
    db.sales_trends.create_index([("date", ASCENDING)])
    db.promotions.create_index([("date", ASCENDING)])
    db.top_products.create_index([("rank", ASCENDING)])

except ConnectionFailure:
    print("Warning: Could not connect to MongoDB. Ensure it is running.")
    db = None
