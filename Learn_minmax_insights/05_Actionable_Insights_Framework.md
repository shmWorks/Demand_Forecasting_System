# 05 — Actionable Insights for 2026+ AI-Driven Retail

## Mental Model

Forecast = inventory plan = cash flow. A 10% RMSLE improvement on GROCERY I (highest volume) saves more than a 50% improvement on LINGERIE (low volume). Prioritize high-volume, high-cannibalization families for model improvement.

## RMSLE Improvement → Business Value

| RMSLE Reduction | Inventory Impact | Revenue Impact (Est.) |
|-----------------|------------------|-----------------------|
| 0.01 | Reduce overstock by ~1-2% | Save ~0.5-1% of COGS |
| 0.05 | Reduce overstock by ~5-8% | Save ~2.5-4% of COGS |
| 0.10 | Reduce both overstock + stockout | Save ~5-8% of COGS + lost sales |

For Favorita (54 stores, ~$1B annual revenue):
- 0.01 RMSLE improvement ≈ $5-10M annual savings in reduced waste + recovered sales

## Prioritization Matrix

```mermaid
graph TD
    subgraph "Impact / Feasibility Matrix"
        H1[{"family": "GROCERY I", "volume": "HIGH", "cannibal": "HIGH", "priority": "P1"}]
        H2[{"family": "BEVERAGES", "volume": "HIGH", "cannibal": "MEDIUM", "priority": "P1"}]
        H3[{"family": "PRODUCE", "volume": "HIGH", "cannibal": "MEDIUM", "priority": "P1"}]
        M1[{"family": "DAIRY", "volume": "MEDIUM", "cannibal": "HIGH", "priority": "P2"}]
        M2[{"family": "MEATS", "volume": "MEDIUM", "cannibal": "HIGH", "priority": "P2"}]
        L1[{"family": "MAGAZINES", "volume": "LOW", "cannibal": "MEDIUM", "priority": "P3"}]
    end
```

**P1** (Improve forecast accuracy first):
- GROCERY I, BEVERAGES, PRODUCE — highest revenue impact
- XGBoost/LGBM with lag_365d feature for yearly seasonality
- Add external features: weather, local events, competitor pricing

**P2** (Cannibalization-aware promotion planning):
- DAIRY, MEATS, FROZEN FOODS — cannibal pairs with PLAYERS AND ELECTRONICS, SCHOOL AND OFFICE SUPPLIES
- Use conflict matrix from cannibalization analysis to schedule non-overlapping promos
- Pre-position inventory for non-promoted cannibal pair

**P3** (Maintain baseline):
- MAGAZINES, HARDWARE, HOME APPLIANCES — low volume, seasonal
- SeasonalNaive sufficient; no model improvement investment warranted

## Cannibalization Conflict Matrix (From 1054 Pairs)

```python
# Build conflict matrix from cannibal_pairs DataFrame
conflict = cannibal_pairs.pivot_table(index='family_i', columns='family_j', values='r', aggfunc='min')
# Minimum correlation across all stores for each pair
# If min_r < -0.7 → never promote simultaneously

# Strategy: Generate promo calendar respecting conflict matrix
def safe_promo_schedule(promo_candidates, conflict_matrix, threshold=-0.7):
    schedule = []
    for candidate in sorted(promo_candidates, key=lambda x: -x.volume):
        conflicts = conflict_matrix.loc[candidate.family]
        active_conflicts = [c for c in schedule if conflicts.get(c.family, 0) < threshold]
        if not active_conflicts:
            schedule.append(candidate)
    return schedule
```

## 2026+ AI Integration Roadmap

| Phase | Action | Expected Lift |
|-------|--------|--------------|
| Q1 | Deploy XGBoost + Optuna as production model | RMSLE −0.03 to −0.05 |
| Q2 | Add cannibalization-aware promo scheduling | Reduce promo waste 5-10% |
| Q3 | Real-time inventory alerts (stockout prediction) | Recover 2-3% lost sales |
| Q4 | External data integration (weather, events) | RMSLE −0.02 incremental |

## Key Metrics to Track

| Metric | Target | Why |
|--------|--------|-----|
| RMSLE | < 0.45 | Kaggle competition threshold for "good" |
| RMSLE by family | GROCERY I < 0.30 | Highest volume, most savings |
| Cannibalization coverage | > 80% of high-volume pairs mapped | Prevents promo conflicts |
| Mean residual bias | < 5% of mean actual | Check for systematic over/under-forecast |
| SHAP top-5 coverage | ≥ 1 promo feature AND ≥ 1 temporal | Validates model reasoning |

## From Numbers to Decisions

```
RMSLE = 0.45 (current) → 0.40 (target)
= 11% relative improvement
= ~$5-10M annual savings for Favorita scale
= $92K-$185K per store per year
= 1 data scientist-year of improvement effort
ROI: Positive from month 1 of deployment
```

## Anti-Patterns to Avoid

1. **Don't optimize MAPE over RMSLE**: MAPE explodes for small actuals. RMSLE is the business-aligned metric.
2. **Don't remove zero-sales rows**: They carry signal (store closed, item not stocked, holiday effect).
3. **Don't use KFold shuffle for CV**: Temporal data requires TimeSeriesSplit. Shuffle leaks future into past.
4. **Don't hardcode paths**: Always use `config.py` constants. Production deployments run on different machines.
5. **Don't skip seed=42**: Non-deterministic models make debugging impossible and audit impossible.
