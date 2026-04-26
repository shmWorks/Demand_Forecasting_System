from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from functools import wraps
import jwt
import datetime
import bcrypt
import os

from dashboard.db import db
from dashboard.models import UserRegister, UserLogin, DashboardMetrics, SalesTrendPoint, PromotionImpactPoint, TopProduct, Recommendation
from pydantic import ValidationError

app = Flask(__name__)
CORS(app)

SECRET_KEY = os.getenv("SECRET_KEY", "brutally-simple-secret")

# --- Error Handling (RFC-7807 minimal) ---
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"type": "about:blank", "title": "Bad Request", "status": 400, "detail": str(e)}), 400

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({"type": "about:blank", "title": "Unauthorized", "status": 401, "detail": str(e)}), 401

@app.errorhandler(404)
def not_found(e):
    return jsonify({"type": "about:blank", "title": "Not Found", "status": 404, "detail": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"type": "about:blank", "title": "Internal Server Error", "status": 500, "detail": str(e)}), 500

# --- JWT Middleware ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith("Bearer "):
            return unauthorized("Missing or invalid token")

        token = token.split(" ")[1]
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = db.users.find_one({"username": data["username"]})
            if not current_user:
                return unauthorized("User not found")
        except Exception as e:
            return unauthorized(f"Token is invalid: {str(e)}")

        return f(current_user, *args, **kwargs)
    return decorated

# --- Routes: Frontend ---
@app.route("/")
def index():
    return render_template("index.html")

# --- Routes: API Auth ---
@app.route("/api/auth/register", methods=["POST"])
def register():
    try:
        data = UserRegister(**request.json)
    except ValidationError as e:
        return bad_request(e.json())

    if db.users.find_one({"username": data.username}):
        return bad_request("Username already exists")

    hashed_pw = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())

    db.users.insert_one({
        "username": data.username,
        "password": hashed_pw
    })

    return jsonify({"message": "User created successfully"}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    try:
        data = UserLogin(**request.json)
    except ValidationError as e:
        return bad_request(e.json())

    user = db.users.find_one({"username": data.username})
    if not user or not bcrypt.checkpw(data.password.encode('utf-8'), user["password"]):
        return unauthorized("Invalid credentials")

    token = jwt.encode({
        "username": user["username"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256")

    return jsonify({"token": token})

# --- Routes: API Dashboard ---
@app.route("/api/dashboard/overview", methods=["GET"])
@token_required
def get_overview(current_user):
    metrics = db.metrics.find_one({}, sort=[("date", -1)], projection={"_id": 0})
    if not metrics:
        return jsonify({"total_sales": 0, "num_products": 0, "active_promotions": 0, "forecast_accuracy": 0})

    # Validate and serialize using Pydantic
    try:
        validated = DashboardMetrics(**metrics)
        return jsonify(validated.model_dump())
    except ValidationError as e:
        return server_error(e.json())

@app.route("/api/dashboard/trends", methods=["GET"])
@token_required
def get_trends(current_user):
    time_filter = request.args.get("filter", "monthly") # week, month, year

    limit = 30
    if time_filter == "weekly":
        limit = 7
    elif time_filter == "yearly":
        limit = 365

    trends = list(db.sales_trends.find({}, projection={"_id": 0}).sort("date", -1).limit(limit))
    trends.reverse() # Chronological order

    try:
        validated_trends = [SalesTrendPoint(**t).model_dump() for t in trends]
        return jsonify(validated_trends)
    except ValidationError as e:
        return server_error(e.json())

@app.route("/api/dashboard/promotions", methods=["GET"])
@token_required
def get_promotions(current_user):
    promos = list(db.promotions.find({}, projection={"_id": 0}).sort("date", -1).limit(14))
    promos.reverse()

    try:
        validated_promos = [PromotionImpactPoint(**p).model_dump() for p in promos]
        return jsonify(validated_promos)
    except ValidationError as e:
        return server_error(e.json())

@app.route("/api/dashboard/top-products", methods=["GET"])
@token_required
def get_top_products(current_user):
    products = list(db.top_products.find({}, projection={"_id": 0}).sort("rank", 1).limit(5))

    try:
        validated_products = [TopProduct(**p).model_dump() for p in products]
        return jsonify(validated_products)
    except ValidationError as e:
        return server_error(e.json())

@app.route("/api/dashboard/recommendations", methods=["GET"])
@token_required
def get_recommendations(current_user):
    recs = list(db.recommendations.find({}, projection={"_id": 0}))

    try:
        validated_recs = [Recommendation(**r).model_dump() for r in recs]
        return jsonify(validated_recs)
    except ValidationError as e:
        return server_error(e.json())

if __name__ == "__main__":
    app.run(debug=True, port=5000)
