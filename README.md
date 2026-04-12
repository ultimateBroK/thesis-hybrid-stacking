# Ứng dụng mô hình Hybrid Stacking dự báo tín hiệu giao dịch CFD Vàng (XAU/USD)

> Đồ án tốt nghiệp — Trường Đại học Thuỷ Lợi, Khoa Công nghệ Thông tin

| Thông tin | Chi tiết |
|-----------|----------|
| **Sinh viên** | Nguyễn Đức Hiếu — 63CNTT.VA — 2151061192 |
| **Giáo viên hướng dẫn** | Hoàng Quốc Dũng |
| **Khung thời gian** | H1 (1 giờ) |
| **Dải dữ liệu** | 01/2018 – 03/2026 |

---

## Tổng quan

Xây dựng pipeline end-to-end dự báo tín hiệu giao dịch **CFD Vàng (XAU/USD)** trên khung H1 bằng kiến trúc **Hybrid Stacking (LSTM + LightGBM)**. Hai mô hình nền dự báo độc lập; mô hình tầng trên học cách kết hợp xác suất đầu ra để phân loại 3 nhãn: **mua (+1), trung tính (0), bán (-1)**.

```
Tầng dưới: LSTM (chuỗi OHLCV 120 nến)  ────┐
                                           ├─> Xác suất 3 lớp ─┐
Tầng dưới: LightGBM (đặc trưng kỹ thuật) ──┘                   ├─> Tầng trên → 3 nhãn
                                                               └─> OOF theo thời gian + Purging/Embargo
```

---

## Mục tiêu chính

1. Thu thập và chuẩn hóa dữ liệu CFD Vàng H1 (01/2018 – 03/2026)
2. Làm sạch dữ liệu: lọc nến bất thường, bỏ cuối tuần, xử lý gap phiên
3. Chia tập theo thời gian (không xáo trộn): Train (2018–2022) / Val (2023) / OOS (2024–03/2026)
4. Ngăn rò rỉ dữ liệu bằng Purging + Embargo
5. Xây dựng 20+ đặc trưng kỹ thuật + định lượng bằng Feature Importance / SHAP
6. Huấn luyện mô hình LSTM nền (chuỗi 120 nến OHLCV)
7. Huấn luyện mô hình LightGBM nền (Optuna tuning, xử lý mất cân bằng)
8. Xây dựng Hybrid Stacking với meta-learner
9. Giải thích mô hình (SHAP) + Backtest một lần trên OOS

---

## Kiến trúc pipeline

```
Raw Tick Data (Dukascopy)
        │
        ▼
  OHLCV H1 (gộp tick → nến)
        │
        ▼
  Làm sạch & Lọc nhiễu (ATR spike filter, bỏ cuối tuần, DST-aware day roll)
        │
        ▼
  Feature Engineering (EMA, RSI, MACD, ATR, Pivot, Spread, Session)
        │
        ▼
  Gán nhãn Triple Barrier (TP=1.5×ATR, SL=1.5×ATR, h=20)
        │
        ▼
  Chia tập theo thời gian + Purging + Embargo (25 nến)
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
  LSTM (120-bar OHLCV)  LightGBM (features)    │
        │                  │                  │
        └──── Xác suất ────┘                  │
                    │                         │
                    ▼                         │
            Meta-Learner (Stacking)           │
                    │                         │
                    ▼                         │
            Dự báo 3 nhãn                     │
                    │                         │
                    ▼                         │
            Backtest OOS ◄────────────────────┘
                    │
                    ▼
            Báo cáo + Figures
```

---

## Cấu trúc dự án

```
thesis/
├── main.py                     # Entry point → thesis.pipeline.runner.run_thesis_workflow()
├── config.toml                 # Cấu hình trung tâm (override qua env THESIS_*)
├── src/
│   └── thesis/
│       ├── __init__.py
│       ├── config/             # Config loader (TOML + env override)
│       ├── data/               # Load/validate/cache OHLCV, splits
│       ├── features/           # Feature engineering (EMA, RSI, MACD, ATR, Pivot, Spread, Session)
│       ├── labels/             # Triple Barrier labeling
│       ├── splitting/          # Time-based split + Purging + Embargo
│       ├── models/
│       │   ├── baseline/       # LightGBM (Optuna tuning, class weighting)
│       │   ├── lstm/           # PyTorch LSTM (CPU-first)
│       │   └── stacking/       # Hybrid Stacking meta-learner
│       ├── backtest/           # CFD backtest engine (costs, sizing, executor, metrics)
│       ├── reporting/          # Markdown/JSON reports + figure generation
│       └── pipeline/           # Workflow runner (stage orchestration)
├── figures/                    # Output figures (11+ charts)
├── data/
│   └── raw/XAUUSD/             # Dữ liệu CFD Vàng H1 (01/2018–03/2026)
├── docs/
│   └── thesis_proposal_summary.md
└── README.md
```

---

## Cấu hình trung tâm (`config.toml`)

| Nhóm | Tham số chính | Mô tả |
|------|--------------|-------|
| **data** | `market_tz`, `day_roll_hour`, `timeframe` | Timezone thị trường, giờ chuyển ngày (DST-aware), khung thời gian |
| **features** | Danh sách indicator + params | EMA(34,89), RSI(14), MACD(12,26,9), ATR(14), Pivot, Spread, Session |
| **labels** | `tp_atr_mult`, `sl_atr_mult`, `horizon`, `atr_coeff` | Triple Barrier: TP=1.5×ATR, SL=1.5×ATR, h=20 |
| **splitting** | `train_end`, `val_end`, `purge_bars`, `embargo_bars` | Mốc chia tập + số nến purge/embargo |
| **models.tree** | LightGBM params, Optuna settings | `num_leaves`, `learning_rate`, `n_estimators`, class weighting |
| **models.lstm** | Sequence length, hidden size, layers, dropout, epochs | `seq_len=120`, early stopping |
| **models.stacking** | Meta-learner type, CV folds | OOF generation theo thời gian |
| **backtest.cfd** | `spread`, `slippage`, `commission`, `leverage`, `swap_rate` | Chi phí giao dịch thực tế |
| **reporting** | Output formats, figure list | Markdown/JSON, 11+ figures |
| **workflow** | `seed`, `cache_dir`, `reuse_base_models` | Reproducibility + cache strategy |
| **paths** | `data_dir`, `artifact_dir`, `figure_dir` | Vị trí lưu artifacts |

Override qua biến môi trường: `THESIS_<SECTION>__<KEY>` (ví dụ `THESIS_LABELS__HORIZON=15`).

---

## Dữ liệu & Artifacts

### Dữ liệu đầu vào
- **Nguồn:** Dukascopy tick data → gộp thành OHLCV H1
- **Dải thời gian:** 01/2018 – 03/2026
- **Số nến sau làm sạch:** ~48.000 – 52.000

### Artifacts tạo ra trong pipeline

| Artifact | Stage tạo | Stage tiêu thụ | Định dạng | Vị trí (Session-based) |
|----------|----------|---------------|-----------|------------------------|
| `ohlcv.parquet` | Data preparation | Features, Labels | Parquet | `data/processed/` (global cache) |
| `features.parquet` | Feature engineering | Labels, Models | Parquet | `data/processed/` (global cache) |
| `labels.parquet` | Labeling | Splitting, Models | Parquet | `data/processed/` (global cache) |
| `train.parquet` / `val.parquet` / `test.parquet` | Splitting | Models | Parquet | `data/processed/` (global cache) |
| `lightgbm_model.pkl` | Baseline training | Stacking | Pickle | `results/{session}/models/` |
| `lstm_model.pt` | LSTM training | Stacking | PyTorch | `results/{session}/models/` |
| `stacking_meta_learner.pkl` | Stacking training | Backtest, Reporting | Pickle | `results/{session}/models/` |
| `predictions.parquet` | Stacking / Inference | Backtest, Reporting | Parquet | `results/{session}/predictions/` |
| `backtest_results.json` | Backtest | Reporting | JSON | `results/{session}/backtest/` |
| `thesis_report.md` / `thesis_report.json` | Reporting | — | Markdown/JSON | `results/{session}/reports/` |
| `figures/*.png` | Reporting | — | PNG | `results/{session}/reports/` |
| `pipeline.log` | All stages | Debugging | Text | `results/{session}/logs/` |

> **Lưu ý:** Mỗi lần chạy pipeline tạo một **session** riêng biệt trong `results/XAUUSD_H1_YYYYMMDD_HHMMSS/`.  
> Symlink `results/latest/` luôn trỏ đến session mới nhất để dễ truy cập.

### Cơ chế cache
- Mỗi stage kiểm tra artifact đầu ra đã tồn tại chưa
- Nếu thiếu artifact → tự chạy stage phụ thuộc (ví dụ: thiếu splits → chạy prepare-data)
- `reuse_base_models=true` → stacking tái dùng baseline/LSTM đã huấn luyện

---

## Rủi ro kỹ thuật & Checklist

### 1. Data Leakage (rò rỉ dữ liệu)
- [ ] Split boundaries: train/val/test không xáo trộn theo thời gian
- [ ] Purging: loại mẫu ở vùng giao thoa khi gán nhãn (h=20)
- [ ] Embargo: chèn khoảng trống 10–25 nến giữa các tập
- [ ] Feature lookahead: đặc trưng chỉ dùng dữ liệu quá khứ (EMA, RSI, MACD...)
- [ ] LSTM window: kiểm tra cửa sổ 120 nến ở ranh giới tập

### 2. Timezone & Day Roll
- [ ] Timestamp chuẩn hóa UTC
- [ ] Trading day neo theo `market_tz` (America/New_York) với `day_roll_hour=17:00`
- [ ] Xử lý DST transition đúng cách (không chỉ dùng UTC)
- [ ] Session features (London/New York/Tokyo) tính theo timezone thị trường

### 3. Class Imbalance
- [ ] Kiểm tra phân phối 3 lớp (+1, 0, -1)
- [ ] Lớp 0 (trung tính) thường chiếm đa số → downsampling hoặc class weighting
- [ ] Đánh giá bằng F1 Macro (không chỉ accuracy)
- [ ] Theo dõi chỉ số từng lớp riêng biệt

### 4. Backtest Realism (giả định CFD)
- [ ] Spread: chênh lệch Bid-Ask
- [ ] Slippage: trượt giá khi khớp lệnh
- [ ] Commission: phí giao dịch
- [ ] Leverage: đòn bẩy
- [ ] Swap rate: phí qua đêm
- [ ] Margin stop-out: gọi vốn bổ sung

### 5. Reproducibility
- [ ] Seed cố định (config `workflow.seed`)
- [ ] Version pinning (requirements.txt / pyproject.toml)
- [ ] Chỉ backtest 1 lần trên OOS (không tuning trên test set)

---

## Đặc trưng kỹ thuật

| Nhóm | Đặc trưng | Mô tả | Câu hỏi thị trường |
|------|-----------|-------|---------------------|
| Xu hướng | EMA(34, 89) | Hướng giá | Giá đang tăng hay giảm? |
| Động lượng | RSI(14) | Sức mạnh nhịp | Đà tăng/giảm mạnh hay yếu? |
| Biến động | ATR(14) | Độ dao động | Thị trường rộng hay hẹp biên? |
| Sức mạnh xu hướng | MACD(12, 26, 9) | Xác nhận xu hướng | Xu hướng còn mạnh không? |
| Phiên giao dịch | Session hour (DST-aware) | Hành vi theo phiên | Giá khác nhau theo phiên? |
| Hỗ trợ/Kháng cự | Pivot Points | Vùng giá quan trọng | Gần vùng quan trọng không? |
| Vi cấu trúc | Spread (Bid-Ask) | Điều kiện khớp lệnh | Chi phí vào lệnh? |

---

## Chiến lược gán nhãn (Triple Barrier)

| Nhãn | Điều kiện | Ý nghĩa |
|------|-----------|---------|
| **+1** | Giá chạm TP trước SL trong cửa sổ h | **Mua** |
| **0** | Không chạm TP/SL đến hết h | **Trung tính** |
| **-1** | Giá chạm SL trước TP trong cửa sổ h | **Bán** |

- **Take Profit:** `Close[t] + 1.5 × ATR`
- **Stop Loss:** `Close[t] - 1.5 × ATR`
- **Horizon:** h = 20 nến
- **Tỷ lệ R:R:** 1:1 (đối xứng)

---

## Kết quả dự kiến

| Thành phần | Metric / Output |
|-----------|-----------------|
| LightGBM | F1 Macro, Accuracy, Feature Importance |
| LSTM | Training history, Probabilities 3 lớp |
| Hybrid Stacking | F1 Macro cao hơn riêng lẻ, SHAP analysis |
| Backtest | Win rate, Total R-multiples, Max Drawdown, Profit Factor, Sharpe, Sortino, Calmar |

---

## Tiến độ thực hiện

| Tuần | Thời gian | Nội dung | Kết quả |
|------|-----------|----------|---------|
| 1 | 23/03 – 29/03 | Xác định yêu cầu, tìm tài liệu | Đề cương kỹ thuật + lịch |
| 2 | 30/03 – 02/04 | Tải & chuẩn hóa + làm sạch dữ liệu | Dữ liệu sạch + báo cáo chất lượng |
| 3 | 03/04 – 07/04 | Tính chỉ báo, gán nhãn, chia tập | Dữ liệu sẵn sàng huấn luyện |
| 4 | 08/04 – 12/04 | Huấn luyện LightGBM | Mô hình + kết quả ban đầu |
| 5 | 13/04 – 20/04 | Huấn luyện LSTM + Stacking | Hybrid Stacking hoàn chỉnh |
| 6 | 21/04 – 30/04 | So sánh mô hình, backtest, SHAP | Số liệu đầy đủ + biểu đồ |
| 7–10 | 01/05 – 28/06 | Viết báo cáo + chuẩn bị thuyết trình | Báo cáo cuối + slide |

---

## Yêu cầu hệ thống

| Yêu cầu | Chi tiết |
|---------|----------|
| Python | 3.13+ |
| RAM | 8GB+ |
| GPU | Tùy chọn (tăng tốc LSTM) |
| Dung lượng | 10GB+ |

### Thư viện chính
- **PyTorch** — LSTM training
- **LightGBM** — Gradient Boosting baseline
- **Optuna** — Hyperparameter tuning
- **SHAP** — Model interpretability
- **Polars** — Xử lý dữ liệu nhanh
- **TA-Lib** — Technical indicators

---

## Hướng phát triển

1. **Ensemble nâng cao:** Weighted voting, Stacking với meta-learner phức tạp hơn
2. **Thêm đặc trưng:** Sentiment analysis, chỉ số vĩ mô (CPI, NFP, lãi suất), Order flow, COT data
3. **Quản lý rủi ro:** Dynamic position sizing, tối ưu danh mục, VaR-based limits
4. **Hệ thống real-time:** WebSocket data, streaming prediction, auto-trading
5. **Đa tài sản:** EUR/USD, GBP/USD, XAG/USD

---

## Tài liệu tham khảo

Xem đầy đủ danh sách 60+ tài liệu tại [`docs/thesis_proposal_summary.md`](docs/thesis_proposal_summary.md).

---

*Cập nhật lần cuối: Tháng 4/2026*
