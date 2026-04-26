from datetime import datetime, timedelta
import random
import bcrypt
from dashboard.db import db

def seed():
    if db is None:
        print("MongoDB not connected. Skipping seed.")
        return

    print("Clearing existing collections...")
    db.users.delete_many({})
    db.metrics.delete_many({})
    db.sales_trends.delete_many({})
    db.promotions.delete_many({})
    db.top_products.delete_many({})
    db.recommendations.delete_many({})

    # 1. User
    print("Seeding admin user...")
    hashed_pw = bcrypt.hashpw(b"admin", bcrypt.gensalt())
    db.users.insert_one({"username": "admin", "password": hashed_pw})

    # 2. Overview Metrics
    print("Seeding metrics...")
    db.metrics.insert_one({
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "total_sales": 1250450.0,
        "num_products": 33,
        "active_promotions": 12,
        "forecast_accuracy": 0.34 # RMSLE
    })

    # 3. Sales Trends (last 30 days)
    print("Seeding trends...")
    today = datetime.utcnow()
    trends = []
    promos = []

    base_sales = 40000
    for i in range(30, -1, -1):
        dt = today - timedelta(days=i)
        dt_str = dt.strftime("%Y-%m-%d")

        # Add seasonality/noise
        day_sales = base_sales + (random.random() * 10000)
        forecast_sales = day_sales + (random.random() * 5000 - 2500)

        trends.append({
            "date": dt_str,
            "sales": round(day_sales, 2),
            "forecast": round(forecast_sales, 2)
        })

        # Promo impact (only need last 14 days)
        if i <= 14:
            normal = day_sales * 0.8
            promo = day_sales * 0.2 + (random.random() * 5000)
            promos.append({
                "date": dt_str,
                "normal_sales": round(normal, 2),
                "promotional_sales": round(promo, 2)
            })

    db.sales_trends.insert_many(trends)
    db.promotions.insert_many(promos)

    # 4. Top Products
    print("Seeding products...")
    products = [
        {"product_id": "P001", "name": "GROCERY I", "sales": 450000, "forecast_growth": 5.2, "rank": 1},
        {"product_id": "P002", "name": "BEVERAGES", "sales": 380000, "forecast_growth": 3.1, "rank": 2},
        {"product_id": "P003", "name": "PRODUCE", "sales": 290000, "forecast_growth": 8.4, "rank": 3},
        {"product_id": "P004", "name": "CLEANING", "sales": 150000, "forecast_growth": 1.5, "rank": 4},
        {"product_id": "P005", "name": "DAIRY", "sales": 120000, "forecast_growth": -0.5, "rank": 5},
    ]
    db.top_products.insert_many(products)

    # 5. Recommendations
    print("Seeding recommendations...")
    recs = [
        {"title": "Increase Stock: PRODUCE", "description": "Forecasts show an 8.4% growth in PRODUCE next week due to seasonal shifts.", "priority": "High"},
        {"title": "Cannibalization Warning", "description": "Promotion on GROCERY I is reducing sales of adjacent SKUs by 12%. Adjust promo cadence.", "priority": "Medium"},
        {"title": "Reduce DAIRY Orders", "description": "Slight negative growth predicted. Reduce safety stock to minimize waste.", "priority": "Low"}
    ]
    db.recommendations.insert_many(recs)

    print("Database seeded successfully! (User: admin / Pass: admin)")

if __name__ == "__main__":
    seed()
