# 00 — Prompt Blindspots Rectified

## What Was Missing

汝 prompt 求 "First Principles deconstruction" + "5 Grokking docs"，然未問：

1. **汝 repository 無 LSTM / Prophet** — prompt 第三章明確列舉 `XGBoost, LSTM, Prophet`，但此 repo 只有 XGBoost + LightGBM (advanced_models.ipynb) + GD_Linear + SeasonalNaive (baseline)。教 LSTM/Prophet 為虛構。

2. **汝未問 RMSLE 公式推導** — 只有「為什麼選 RMSLE」，無「RMSLE = sqrt(mean((log1p(pred) - log1p(actual))²)) 從何推來」。需從 maximum likelihood 推導。

3. **汝未問 SHAP fallback 風險** — `evaluation.py:L116-L128` 顯示 TreeExplainer 失敗後自動 fallback 到 KernelExplainer。此 fallback 慢 10x 且結果不穩定。

4. **汝未問 GD_Linear 的 log1p 假設** — 模型在 log1p 空間訓練，但 `predict()` 直接做 `X @ theta` 而未逆變換。呼叫者需自行 `expm1()`。此 API 契約文件未說明。

5. **汝未問 Temporal Holdout 為何 15 days** — `preprocessing.py:L189` `holdout_days=15` 硬編碼。SPEC.md 只說「Train < 2017-08-16, Test = 2017-08-16 to 2017-08-31」。15 days = Aug 16-31 inclusive。此選擇的優缺點未探討。

6. **汝未問 Optuna timeout 風險** — `advanced_models.ipynb` 的 Optuna `timeout=600` (10 分鐘) 但無 early stopping。若 50 trials 全部慢於 12 分鐘，則 timeout 後無完整結果。

7. **汝未問 Zero-Inflation 建模** — 40-60% rows 為零銷售。現有模型 (GD_Linear, XGBoost, LightGBM) 均為連續輸出，無法建模 "P(sales=0)" 與 "E(sales|sales>0)" 的混合分佈。忽視 zero-inflation 導致預測偏低。

8. **汝未問 Cannibalization 的 p-value** — `find_cannibal_pairs()` 只輸出 correlation r，無顯著性檢驗。SPEC.md §B 要求 `≥3 cannibal pairs with r, p-value`，但 report 只有 r。

9. **汝未問 Oil Price 的交易日錯位** — oil.csv 無週末數據，但 sales 有。`merge_datasets()` 後 oil column 週末為 NaN，經 `clean_oil_prices()` ffill 填充。問題：週六用週五油價合理，但 Ecuador 市場週六不開市，此填充引入了假相關。

10. **汝未問 Holiday `transferred` 排除邏輯** — `preprocessing.py:L141` 排除 `transferred==True` holidays，`features.py:L104` 亦然。**但 transferred holidays 才是 Ecuador 特有的「挪假效應」** — 員工在 holiday 當日工作而後補休。此效應對 sales 有真實影響，被過濾掉了。

## Rectified: What The Prompt Should Have Asked

| # | Missing Question | Why Critical | Source |
|---|----------------|-------------|--------|
| 1 | "此 repo 有哪些模型？與 prompt 列舉是否匹配？" | 防止教 LSTM/Prophet — repo 不存在 | advanced_models.ipynb |
| 2 | "RMSLE 從最大似然估計推導為何？" | 理解 metric 假設，判斷是否適用於 zero-inflated 分佈 | evaluation.py:L44 |
| 3 | "GD_Linear API 契約：predict() 輸出什麼空間？" | log1p 空間。呼叫者需自行逆變換。文件未說明 | models.py:L167-L181 |
| 4 | "Zero-inflated 分佈如何建模？" | 40-60% 零值。現有模型輸出連續值，低估零時期。需 two-stage model 或 Tweedie | - |
| 5 | "Holiday transferred 排除是否正確？" | transferred=True = 挪假效應，可能比 national holiday 更重要 | preprocessing.py:L141 |
| 6 | "Optuna timeout 是否足夠完成搜索？" | 600s timeout 若 trial > 12s 可能中斷搜索 | advanced_models.ipynb |
| 7 | "SHAP KernelExplainer fallback 慢 10x — 如何避免？" | 在小樣本上用 KernelExplainer 可行，但需預期性能下降 | evaluation.py:L124 |
| 8 | "Cannibalization pairs 的 p-value 在哪？" | SPEC 要求 p-value，report 只有 r | outputs/reports/cannibalization_report.md |
| 9 | "Oil 週末填充是否引入假相關？" | 週六 sales 用週五 oil — 落後一日的相關可能是假的 | preprocessing.py:L97 |
| 10 | "Holdout 為何選 15 days？有何替代？" | 15 days = Aug 16-31。替代：rolling 7-day window 更 robust | preprocessing.py:L189 |

## Rectified Document Additions Needed

### Addition to `03_Mathematics_of_Forecasting.md`

**RMSLE 推導（Maximum Likelihood）:**

假設誤差服從 log-normal 分佈：$\epsilon = \log(y_{pred}) - \log(y_{actual}) \sim \mathcal{N}(0, \sigma^2)$

Maximum likelihood 估計 $\sigma^2$：
$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (\log y_{pred} - \log y_{actual})^2$$

RMSLE = $\sqrt{\hat{\sigma}^2}$ = RMSE on log-space

**為何 log-normal 假設合理於 sales：**
- Sales 為非負、right-skewed
- 乘法過程（multiple independent factors）自然產生 log-normal
- 零售需求受多因素影響（天氣、促銷、競爭），複合效應 → log-normal

### Addition to `04_Cannibalization_and_Lift.md`

**p-value 計算缺失：**

汝 report 缺 p-value。以下為補充方法：

```python
from scipy import stats

def find_cannibal_pairs_with_pvalue(df, promo_threshold=0, corr_threshold=-0.35):
    stores = df['store_nbr'].unique()
    pairs = []
    for s in stores:
        df_s = df[df['store_nbr'] == s].copy()
        df_s['sales_resid'] = df_s.groupby('family')['sales'].transform(
            lambda x: x.shift(1).rolling(28, min_periods=1).mean()
        )
        promo_mask = df_s['onpromotion'] > promo_threshold
        df_promo = df_s[promo_mask].pivot_table(index='date', columns='family', values='sales_resid')
        corr = df_promo.corr()
        n = len(df_promo)  # sample size
        
        for i, fam_i in enumerate(corr.columns):
            for fam_j in corr.columns[i+1:]:
                r = corr.loc[fam_i, fam_j]
                if r < corr_threshold:
                    # Transform r to t-statistic
                    # t = r * sqrt(n-2) / sqrt(1-r^2)
                    t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
                    p_value = 2 * stats.t.sf(np.abs(t_stat), df=n-2)
                    pairs.append({'store': s, 'family_i': fam_i, 'family_j': fam_j, 'r': r, 'p_value': p_value})
    return pd.DataFrame(pairs)
```

### Addition to `03_Mathematics_of_Forecasting.md`

**Zero-Inflation Gap（汝從未問）:**

```python
# 現有模型的問題：
# GD_Linear, XGBoost, LightGBM 輸出 E[sales]，但 sales 分佈：
# P(sales) = π * I(sales=0) + (1-π) * LogNormal(μ, σ)
# 其中 π = P(sales=0) ≈ 40-60% for Favorita

# 連續模型預測：
# 預測 = E[sales] = (1-π) * exp(μ + σ²/2)
# 問題：當 π 很高時，E[sales] 趨近於 0，但實際上在非零時期銷量可觀

# 解決方案：Two-Stage Model
# Stage 1: Classifier — P(sales > 0 | features)  [logistic/GBM binary]
# Stage 2: Regressor — E[sales | sales > 0]      [XGBoost/LGBM continuous]
# 最終預測 = P(sales>0) * E[sales | sales>0]

# 此 repo 無此實現 — 為 future work
```

### Addition to `01_System_Architecture.md`

**Holiday transferred 爭議（汝從未問）:**

```python
# preprocessing.py:L141 — 排除 transferred holidays
active_holidays = holidays[holidays["transferred"] == False]
national_dates = set(active_holidays.loc[active_holidays["locale"] == "National", "date"])
df["is_national_holiday"] = df["date"].isin(national_dates).astype(np.int8)

# features.py:L104 — 同一過濾
active = self.holidays[self.holidays["transferred"] == False]
```

**問題：**
- `transferred=True` = holiday 從原日期挪到工作日（原 holiday 變工作日）
- Ecuador挪假模式：聖誕節 (Dec 25) → 若 Dec 25 是週六，則 Dec 24 (Fri) 變假日
- **transferred day 的銷售模式其實比原 holiday 更異常**
- 過濾掉 transferred = 失去最重要的挪假信號

**正確做法：**
```python
# 兩特徵：
df["is_national_holiday"] = df["date"].isin(national_dates).astype(np.int8)
df["is_transferred_holiday"] = df["date"].isin(transferred_dates).astype(np.int8)
# 或計算"actual holiday type"：若原 date 是 holiday，返回 locale
```
