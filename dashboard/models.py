from pydantic import BaseModel, Field
from typing import List, Optional

# --- Auth Models ---
class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str

# --- Dashboard Models ---
class DashboardMetrics(BaseModel):
    total_sales: float
    num_products: int
    active_promotions: int
    forecast_accuracy: float # e.g. RMSLE

class SalesTrendPoint(BaseModel):
    date: str
    sales: float
    forecast: Optional[float] = None

class PromotionImpactPoint(BaseModel):
    date: str
    normal_sales: float
    promotional_sales: float

class TopProduct(BaseModel):
    product_id: str
    name: str
    sales: float
    forecast_growth: float
    rank: int

class Recommendation(BaseModel):
    title: str
    description: str
    priority: str # High, Medium, Low
