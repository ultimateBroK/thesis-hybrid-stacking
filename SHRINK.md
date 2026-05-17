## Chốt hướng tối ưu

Tôi khuyên bạn chuyển từ **6-stage production-style pipeline** sang **4-stage academic ML pipeline**:

```text
Stage 1: Dataset Preparation
Stage 2: Feature & Label Engineering
Stage 3: Model Training & Evaluation
Stage 4: Result Analysis & Thesis Report
```

Backtest không nên là một stage chính nữa. Nên đưa nó xuống thành:

```text
optional_demo/backtest_demo.py
```

hoặc một section nhỏ trong report: **“Ứng dụng minh họa tín hiệu vào giao dịch”**.

Lý do: đồ án CNTT cần chứng minh bạn hiểu **quy trình ML, thiết kế dữ liệu, mô hình, validation, metric, so sánh baseline**, không cần chứng minh hệ thống giao dịch có lời.

---

## Kết cấu stage nên đổi thành thế này

### Stage 1 — Dataset Preparation

Gộp từ `stage_1_data`.

Nhiệm vụ chỉ nên là:

```text
Raw XAU/USD data
→ resample/clean thành OHLCV H1
→ kiểm tra thiếu dữ liệu cơ bản
→ lưu dataset chuẩn
```

Giữ lại:

* đọc dữ liệu;
* chuẩn hóa timestamp;
* tạo OHLCV H1;
* basic missing/gap check.

Nên bỏ hoặc giảm:

* outlier analysis quá sâu;
* nhiều data quality report phụ;
* session path/caching quá production-style.

Output nên đơn giản:

```text
data/processed/xauusd_h1.parquet
reports/data_summary.json
```

Hiện architecture đang mô tả Stage 1 là raw ticks → OHLCV, có `ohlcv.parquet` và data quality json. Phần này đúng, chỉ cần làm gọn. ([GitHub][2])

---

### Stage 2 — Feature & Label Engineering

Gộp `stage_2_features` và `stage_3_labels`.

Đây là điều chỉnh quan trọng nhất.

Hiện tại bạn tách features và triple-barrier labels thành hai stage riêng. Về production thì ổn, nhưng với đồ án CNTT, nên gộp thành một stage vì cả hai đều thuộc **xây dựng bộ dữ liệu học máy**.

```text
OHLCV H1
→ technical features
→ triple-barrier labels
→ final ML dataset
```

Giữ lại feature groups chính:

* trend: EMA, ADX;
* momentum: return, RSI, MACD;
* volatility: ATR;
* position: price position, VWAP/pivot nếu muốn;
* session: Asia/London/NY nếu bạn giải thích được.

Repo hiện có khoảng 21 causal features, chia theo trend, momentum, volatility, position, candle, session. ([GitHub][2]) Cấu trúc này ổn, nhưng nên **chốt 12–16 feature dễ bảo vệ**, không nên ham nhiều.

Nên bỏ hoặc giảm:

* các feature “trông có vẻ trading quá” nhưng khó giải thích;
* feature quality validation quá nhiều;
* Pandera schema nếu nó làm code rối;
* profitability check trong labeling.

Triple-barrier label vẫn nên giữ vì nó làm đề tài có tính học thuật hơn so với “next close up/down”. Hiện Stage 3 của bạn dùng upper/lower barrier theo ATR và horizon forward; nhãn gồm Long, Hold, Short, Censored. ([GitHub][2]) Đây là phần đáng giữ.

Nhưng tôi khuyên đổi tên từ:

```text
Stage 3: Label Generation
```

thành:

```text
Label Construction for Supervised Learning
```

Nghe học thuật hơn, ít giống bot trade hơn.

Output:

```text
data/modeling/ml_dataset.parquet
reports/label_distribution.json
reports/feature_list.json
```

---

### Stage 3 — Model Training & Evaluation

Gộp trọng tâm từ `stage_4_training`.

Đây phải là trái tim của đồ án.

Pipeline nên là:

```text
Input: ml_dataset.parquet

Baselines:
- Majority Class
- Logistic Regression
- Random Forest
- LightGBM

Main model:
- Hybrid Stacking:
  Logistic Regression + Random Forest + LightGBM
  → Meta Logistic Regression

Validation:
- Chronological split hoặc walk-forward đơn giản

Metrics:
- Accuracy
- Macro F1
- Precision/Recall/F1 per class
- Confusion matrix
```

Hiện repo đang đúng tinh thần này: base learners gồm Logistic Regression, Random Forest, LightGBM; meta learner là Logistic Regression trên class-probability outputs của base learners. ([GitHub][2]) Định nghĩa stacking chính thống cũng là dùng output của các estimator làm input cho final estimator. ([Scikit-learn][3])

Nhưng nên giảm bớt complexity:

Nên giữ:

* walk-forward validation;
* base/model comparison;
* confusion matrix;
* Macro F1;
* per-class metrics;
* feature importance.

Nên cân nhắc bỏ hoặc đưa vào appendix:

* distribution-shift weights;
* sample weights nếu code đang làm rối;
* calibration ECE/Brier nếu chưa cần;
* quá nhiều baseline random/always class nếu report bị loãng;
* nhiều artifact persistence.

Với đồ án, tôi khuyên chỉ giữ 4 mô hình so sánh:

```text
1. Logistic Regression
2. Random Forest
3. LightGBM
4. Hybrid Stacking
```

Thêm Majority Class làm baseline tối thiểu là đủ.

---

### Stage 4 — Result Analysis & Thesis Report

Gộp `stage_6_reporting`, bỏ bớt tính “dashboard”.

Nhiệm vụ:

```text
metrics.json
confusion_matrix.png
model_comparison.csv
feature_importance.png
final_report.md
```

Report nên trả lời 5 câu hỏi:

```text
1. Dữ liệu dùng là gì?
2. Nhãn Short/Hold/Long được tạo như thế nào?
3. Hybrid Stacking hoạt động ra sao?
4. Mô hình có tốt hơn baseline không?
5. Hạn chế và hướng phát triển là gì?
```

Hiện Stage 6 của bạn đang có nhiều section: executive summary, config, data quality, label methodology, validation, classification metrics, calibration. ([GitHub][2]) Cái này tốt, nhưng nên cắt xuống thành report học thuật gọn, tránh cảm giác “tool quá to so với đồ án”.

---

## Stage nên bỏ / hạ cấp

### Nên hạ cấp Stage 5 Backtest

Hiện backtest đang có confidence filter, position sizing, drawdown cutoff, daily loss limit, cooldown, ATR stop-loss/take-profit, spread, slippage, commission. ([GitHub][2]) Đây là phần dễ làm hội đồng lệch hướng hỏi về trading/risk management thay vì ML.

Nên đổi từ:

```text
src/thesis/stage_5_backtest/
```

thành:

```text
src/thesis/demo/backtest_demo.py
```

Trong thesis chỉ ghi:

> Backtest được sử dụng như minh họa ứng dụng đầu ra mô hình, không phải tiêu chí chính để đánh giá chất lượng mô hình.

Không nên để backtest là stage chính trong pipeline bảo vệ.

---

## Cấu trúc thư mục tôi khuyên dùng

```text
src/thesis/
  data/
    prepare_dataset.py

  dataset/
    build_features.py
    build_labels.py
    build_ml_dataset.py

  models/
    baselines.py
    stacking.py
    train.py
    evaluate.py

  reporting/
    metrics.py
    plots.py
    report.py

  demo/
    backtest_demo.py

  shared/
    config.py
    constants.py
    utils.py

main.py
config.toml
```

Nhìn sẽ “CNTT + ML” hơn nhiều so với:

```text
stage_1_data
stage_2_features
stage_3_labels
stage_4_training
stage_5_backtest
stage_6_reporting
```

Tên `stage_*` không sai, nhưng hơi giống production data pipeline. Với thesis, tên theo domain học thuật dễ bảo vệ hơn.

---

## Những thứ nên giữ để “đúng đề tài”

| Thành phần                   |        Giữ không? | Lý do                           |
| ---------------------------- | ----------------: | ------------------------------- |
| XAU/USD H1                   |               Giữ | Đúng bài toán CFD vàng          |
| Short/Hold/Long              |               Giữ | Phù hợp “tín hiệu giao dịch”    |
| Triple-barrier labeling      |               Giữ | Có tính học thuật               |
| Causal features              |               Giữ | Tránh leakage                   |
| Walk-forward validation      |               Giữ | Rất quan trọng với time series  |
| Logistic Regression baseline |               Giữ | Dễ giải thích                   |
| Random Forest                |               Giữ | Đại diện bagging/tree           |
| LightGBM                     |               Giữ | Đại diện boosting               |
| Meta Logistic Regression     |               Giữ | Là lõi stacking                 |
| Backtest phức tạp            |            Hạ cấp | Dễ làm lệch trọng tâm           |
| Dashboard                    |  Bỏ hoặc appendix | Không cần cho bảo vệ ML         |
| Calibration/ECE/Brier        |          Optional | Hay nhưng không bắt buộc        |
| Distribution-shift weights   |          Bỏ trước | Khó bảo vệ nếu hội đồng hỏi sâu |
| Sample uniqueness weights    | Optional/appendix | Học thuật nhưng dễ làm rối      |

---

## Tên đề tài nên giữ như này

**“Ứng dụng mô hình Hybrid Stacking trong dự báo tín hiệu giao dịch CFD vàng”**

Phạm vi nghiên cứu nên ghi rõ:

> Đề tài tập trung vào xây dựng và đánh giá mô hình phân loại tín hiệu Short/Hold/Long cho XAU/USD khung H1. Kết quả được đánh giá chủ yếu bằng các chỉ số Machine Learning như Accuracy, Macro F1, Precision, Recall và Confusion Matrix. Backtest nếu có chỉ đóng vai trò minh họa ứng dụng, không phải bằng chứng chính về lợi nhuận giao dịch.

Câu này rất quan trọng. Nó khóa phạm vi, tránh bị hỏi lan man về chiến lược giao dịch.

---

## Việc nên làm ngay theo thứ tự

1. **Đổi narrative trước, chưa cần sửa code ngay**
   README/docs phải nói rõ: đây là supervised ML classification thesis, không phải trading bot.

2. **Ẩn hoặc hạ cấp backtest**
   Không để backtest đứng ngang hàng với training/evaluation.

3. **Gộp Stage 2 + Stage 3 trong docs**
   Code có thể vẫn tách file, nhưng trong báo cáo gọi chung là “Dataset Construction”.

4. **Giảm report metric**
   Chỉ giữ bảng so sánh model, confusion matrix, Macro F1, per-class F1, feature importance.

5. **Chốt mô hình chính**
   Hybrid Stacking = LR + RF + LightGBM → Meta LR. Không thêm GRU/LSTM nữa.

Một câu chốt: **đừng làm đồ án thành “hệ thống giao dịch vàng”; hãy làm nó thành “nghiên cứu ML phân loại tín hiệu giao dịch vàng”.** Đây là hướng dễ bảo vệ hơn, đúng ngành CNTT hơn, và phù hợp codebase hiện tại nhất.

[1]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-refactor/README.md "raw.githubusercontent.com"
[2]: https://github.com/ultimateBroK/thesis-hybrid/blob/hybrid-refactor/docs/ARCHITECTURE.md "thesis-hybrid/docs/ARCHITECTURE.md at hybrid-refactor · ultimateBroK/thesis-hybrid · GitHub"
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html?utm_source=chatgpt.com "StackingClassifier"
