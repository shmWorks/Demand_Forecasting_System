### **Retail-IQ: High-Performance Engineering Guide**

#### **1. Caching Strategy**
*   **Best Practice**: Prefer `lru_cache` (pure functions) or Redis (distributed).
*   **Warning**: **Avoid pickle-based caches** (`joblib`, `diskcache`) due to **CVE-2025-69872** (RCE). Use JSON/Msgpack or SafeTensors instead.

#### **2. Concurrency (Process vs. Thread)**
*   **CPU-bound**: `ProcessPoolExecutor` (Bypass GIL).
*   **I/O-bound**: `ThreadPoolExecutor` (Overlap tasks).
*   **NoGIL (Python 3.14+)**: Threading now enables 10x speedup for CPU tasks. Retrofit existing `ThreadPoolExecutor` code.
*   **Critical Rules**: 
    *   Windows: Use `if __name__ == '__main__':`.
    *   Pickling: Avoid `lambda` in pools; use named functions.

#### **3. Data & Memory Optimization**
*   **Precision**: Use `float32` (2x memory reduction, ~1.5x speedup).
*   **Pandas/NumPy**: Use `itertuples` (10x faster than `iterrows`), `np.asarray` (views), and preallocate arrays.
*   **Scaling**: Use **Polars** for large joins/groupbys (5-50x faster than Pandas).

#### **4. ML & Hardware Acceleration**
*   **Training (XGB/LGBM)**: Set `n_jobs=-1`. If parallelizing models, set `os.environ['OMP_NUM_THREADS'] = '1'` per model to prevent contention.
*   **SHAP**: Use `GPUTreeShap` for 10-340x speedups.
*   **DataLoader (PyTorch)**: Use `pin_memory=True` and `persistent_workers=True`.

#### **5. Decision Checklist**
| Scenario | Recommended Strategy |
| :--- | :--- |
| **CPU + Small Data** | `ThreadPoolExecutor` (NoGIL) |
| **CPU + Large Data** | `ProcessPoolExecutor` + Shared Mem |
| **I/O Bound** | `ThreadPoolExecutor` |
| **Feature Compute** | `ProcessPoolExecutor` (Chunk by store/family) |
| **ML Training** | `n_jobs=-1` |

---
**Confidence level**: 0.99
**Key caveats**: 
*   **Security**: The RCE risk (pickle) is critical; prioritize migrating to `SafeTensors` for ML model/data persistence. 
*   **OS/Env**: Always test `ProcessPoolExecutor` behavior, as fork/spawn defaults vary by OS.