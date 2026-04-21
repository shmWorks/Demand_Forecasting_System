# Literature Review Division Plan (30 Papers)

## 1. Summary Table

| Category | **Person A: The Architect** | **Person B: The Scientist** | **Person C: The Strategist** |
| :--------------------- | :------------------------------------------------------------ | :------------------------------------------------------- | :-------------------------------------------------------- |
| **Core Models** | [1] Survey of Trends<br>[4] Probabilistic GB | [2] Algorithm Comparison<br>[23] Rolling Window Training | [3] Regime-Based GB<br>[30] Spatial-Temporal GBT |
| **Promotion & Market** | [7] Promo Lifecycle | [5] Cannibalization Features<br>[8] Cross-Category SHAP | [6] Counterfactual Lift |
| **Feature & Data** | [10] Quantile Feature Eng.<br>[13] Intermittent Demand | [11] Stockout-Aware Eng.<br>[29] Ecuador Oil Macro | [9] Feature Eng. Guide<br>[12] Zero-Sale Classification |
| **Explainability** | [16] SHAP Interaction | [14] SHAP in Retail Context | [15] SHAP vs LIME Stability |
| **Deep Learning** | [19] Chronos-2 (Foundation)<br>[22] GRU-LightGBM Hybrid | [17] Temporal Fusion Transf.<br>[20] Causal Transformers | [18] Transformer vs Linear<br>[21] XGBoost-LSTM Hybrid |
| **Operations** | [25] Hierarchical Reconciliation<br>[28] Perishable Inventory | [26] Optuna v4 Optimization | [24] Retraining Stability<br>[27] Data Leakage Prevention |

---

## 2. Team Member Assignments

### **Person A: High-Scale Systems & Foundation Modeling**

_Focus: Large-scale systems, new tech, sparse data._

1. **[1] Ahaggach et al.** — Systematic study of sales forecasting trends.
2. **[4] Long et al.** — Scalable probabilistic forecasting with GB trees.
3. **[7] Hewage et al.** — Comprehensive analysis of 3-phase promo lifecycle.
4. **[10] Wen et al.** — Multi-horizon quantile forecasting with temporal FE.
5. **[13] Muşat & Căbuz** — Switch-Hurdle models for intermittent demand.
6. **[16] Kumar et al.** — SHAP feature interaction analysis for supply chains.
7. **[19] Ansari et al.** — Chronos-2: Applying foundation models to forecasting.
8. **[22] Y. Chen et al.** — GRU-LightGBM hybrid for short-shelf-life products.
9. **[25] Wickramasuriya et al.** — Robust hierarchical forecast reconciliation.
10. **[28] Y. Kim et al.** — Linking forecasts to replenishment decisions.

### **Person B: Statistical Rigor & Macro Dynamics**

_Focus: Feature signal, external drivers (Oil), temporal attention._

1. **[2] Santos et al.** — Benchmarking CatBoost, LightGBM, and XGBoost.
2. **[5] van den Berg et al.** — Cross-product cannibalization features.
3. **[8] Huang et al.** — Cross-category demand shifts via SHAP.
4. **[11] Szabłowski** — Stockout-aware FE for global model behaviors.
5. **[14] Borba et al.** — Interpreting retail GB models using SHAP.
6. **[17] Leinonen et al.** — TFT for retail with explanatory variables.
7. **[20] Wang et al.** — Causal-aware multimodal Transformers for supply chain.
8. **[23] Gomes et al.** — Ensemble ensembles with rolling window training.
9. **[26] Akiba et al.** — Optuna v4 for rapid hyperparameter optimization.
10. **[29] Espinoza et al.** — Oil price volatility impact on Ecuador retail.

### **Person C: Strategic Integrity & Insight Generation**

_Focus: Promo lift, spatial relations, model stability/validation._

1. **[3] G. Chen et al.** — Multi-regime demand forecasting (Promo/Normal).
2. **[6] Geurts & Nagaraj** — Measuring true lift, halo, and cannibalization.
3. **[9] Petropoulos et al.** — Practical guide to lag and rolling features.
4. **[12] Farahani et al.** — Model-based approach for zero demand classification.
5. **[15] A. K. Sharma et al.** — Comparative stability of SHAP, LIME, and Attention.
6. **[18] Y. Liu et al.** — Analyzing effectiveness of Transformers vs. Linear.
7. **[21] J. Zhang et al.** — Hybrid XGBoost-LSTM for multi-store data.
8. **[24] X. Long et al.** — Impact of retraining frequency on forecast stability.
9. **[27] Cerqueira et al.** — Quantifying data leakage in time-series splits.
10. **[30] X. Li et al.** — Spatial-temporal GB using store-cluster similarities.

---

## 3. Collaboration Sync Points

Meet at these "Technical Intersection" points:

- **The Promotional Task Force (A-7, B-5, B-8, C-3, C-6):** Finalize definitions for FR-08 (Cannibalization proxy features).
- **The XAI Workshop (A-16, B-14, C-15):** Coordinate consistent SHAP interpretation methodology for FR-07 module.
- **The Data Integrity Circle (A-10, B-11, C-9, C-27):** Ensure feature engineering pipeline (FR-03) follows leakage-prevention best practices.