Đánh giá nhanh: **dự án đã tốt hơn về mặt “bảo vệ đồ án”**, đặc biệt ở chỗ README/docs đã nhấn mạnh: classification metrics là kết quả chính, backtest chỉ là application demo, có walk-forward, purge/embargo, 3-class target, baseline comparison và report rõ ràng. Repo hiện tại ghi rõ primary output là Accuracy/F1/Precision-Recall/Confusion Matrix, còn backtest chỉ là minh họa tín hiệu, không phải bằng chứng lợi nhuận. ([GitHub][1])

Nhưng có một vấn đề lớn cần xử lý trước: **tên đề tài là “Hybrid Stacking”, còn repo hiện tại vẫn đang mô tả “Hybrid GRU + LightGBM” nhiều hơn là stacking đúng nghĩa**. Theo scikit-learn, stacked generalization là dùng đầu ra của các base estimators làm đầu vào cho final estimator/meta-model. ([scikit-learn.org][2]) Còn kiến trúc hiện tại của bạn chủ yếu là **GRU hidden states + static features → LightGBM**, tức gần với **feature-level hybrid / representation fusion** hơn là stacking ensemble chuẩn.

## Ưu tiên 1 — Chốt lại định nghĩa “Hybrid Stacking”

Hiện tại README ghi mô hình là:

```text
GRU reads 48 hours → outputs temporal embedding
LightGBM combines embedding + static indicators → predicts Short/Hold/Long
```

Repo cũng mô tả 4 nhóm so sánh: Naive Direction, LightGBM Static, GRU-only, Hybrid GRU+LightGBM. ([GitHub][1])

Cách này **có tính hybrid**, nhưng chưa thật sự “stacking” theo nghĩa học thuật phổ biến.

Bạn có 2 lựa chọn:

| Lựa chọn                                                                               |                       Nên làm? | Lý do                                                    |
| -------------------------------------------------------------------------------------- | -----------------------------: | -------------------------------------------------------- |
| Giữ GRU hidden states + LightGBM và gọi là Hybrid Stacking                             |                      Không nên | Dễ bị bắt bẻ: đây là feature fusion, không phải stacking |
| Đổi mô hình thành actual stacking: GRU-only proba + LightGBM-static proba → meta-model |                        **Nên** | Khớp tên đề tài, dễ bảo vệ hơn                           |
| Giữ cả hai: Hybrid Fusion là baseline, Hybrid Stacking là final model                  | **Tốt nhất nếu còn thời gian** | Có so sánh học thuật rõ                                  |

Tôi khuyên sửa kiến trúc chính thành:

```text
Base model 1: LightGBM static features → predict_proba
Base model 2: GRU sequence model → predict_proba
Meta model: Logistic Regression / LightGBM nhỏ → final Short/Hold/Long
```

Như vậy bạn có thể nói:

> “Hybrid” vì kết hợp deep sequence model và tree-based model.
> “Stacking” vì meta-model học từ xác suất đầu ra của các base models.

Đây là thay đổi **đáng làm nhất**.

---

## Ưu tiên 2 — Đừng để GRU làm dự án quá nặng

Config mới đã giảm độ phức tạp GRU: hidden size 32, sequence length 24, 1 layer, dropout 0.5, epochs 30, PCA 8. Đây là hướng tốt vì giảm overfitting và chạy nhanh hơn. ([GitHub][3])

Nhưng tôi vẫn thấy GRU đang chiếm quá nhiều “trọng lượng luận văn”. Trong docs, GRU vẫn được trình bày như lõi chính, có focal loss, cosine annealing, PCA, distribution-shift weights, contrastive pretrain. ([GitHub][4])

Với tiêu chí **đơn giản, dễ bảo vệ**, bạn nên hạ vai trò GRU xuống:

```text
GRU = một base learner trong hệ stacking
LightGBM static = base learner chính
Meta-model = nơi tạo Hybrid Stacking final signal
```

Không nên viết như thể GRU là phát minh trung tâm. Hội đồng rất dễ hỏi sâu GRU, focal loss, PCA, hidden state, attention pooling, contrastive pretraining. Nếu bạn không muốn tự đẩy mình vào thế khó, hãy làm cho câu chuyện đơn giản hơn:

```text
Dữ liệu chuỗi → GRU học đặc trưng thời gian
Dữ liệu bảng → LightGBM học đặc trưng kỹ thuật
Meta-model → kết hợp xác suất từ hai nguồn
```

---

## Ưu tiên 3 — Sửa inconsistency giữa README, docs và config

Tôi thấy một số chỗ đang lệch nhau:

| Thành phần        | README/docs nói                    | config hiện tại nói                | Vấn đề         |
| ----------------- | ---------------------------------- | ---------------------------------- | -------------- |
| Data range        | README: Jan 2018 – Apr 2026        | config: start_date 2021-01-01      | Lệch dữ liệu   |
| Walk-forward      | README: 2-year train, 6-month test | config: 1-year train, 2-month test | Lệch mô tả     |
| GRU hidden        | docs có chỗ nói 64                 | config: 32                         | Lệch kiến trúc |
| GRU sequence      | README/docs có chỗ nói 48          | config: 24                         | Lệch input     |
| PCA               | docs có chỗ nói 16                 | config: 8                          | Lệch số chiều  |
| Features          | README nói 21/22/28 tùy chỗ        | config/docs khác nhau              | Dễ bị hỏi      |
| Backtest cooldown | docs nói min_bars_between_trades=6 | config: 18                         | Lệch tham số   |

Đây là lỗi **rất nguy hiểm khi bảo vệ**, không phải vì code sai, mà vì báo cáo thiếu nhất quán.

Việc cần làm ngay:

```text
1. Chọn một config chính thức.
2. Chạy lại workflow.
3. Snapshot config.
4. Đồng bộ README.md, docs/ARCHITECTURE.md, docs/CONFIGURATION.md, docs/TUNING.md.
5. Trong báo cáo chỉ dùng đúng một bộ số.
```

Tôi khuyên đặt tên rõ:

```text
Official thesis profile:
- Data: 2021-01-01 → 2026-04-30
- Timeframe: H1
- Labels: 3-class Triple Barrier
- Horizon: 24 bars
- Validation: sliding walk-forward
- Train window: 1 year
- Test window: 2 months
- Purge: 48 bars
- Embargo: 50 bars
- Final model: Hybrid Stacking
```

---

## Ưu tiên 4 — Giữ 3-class, nhưng thêm bảng label distribution theo window

Bạn đã giữ 3 class: Short / Hold / Long. Đây là đúng hướng. Repo hiện mô tả labels bằng Triple Barrier và class Short/Hold/Long. ([GitHub][1])

Nhưng bắt buộc phải thêm bảng:

```text
window_id | train_short | train_hold | train_long | test_short | test_hold | test_long
```

Lý do: với tài chính, label distribution drift là chuyện rất thường gặp. Docs của bạn cũng đã nhắc tới label drift và distribution-shift weights. ([GitHub][5]) Nhưng khi bảo vệ, nếu không có bảng phân phối nhãn theo window thì phần này hơi “nói miệng”.

Metric cũng nên ưu tiên:

```text
Macro F1
Balanced Accuracy
Per-class Precision/Recall
Confusion Matrix
```

Balanced accuracy đặc biệt hợp lý vì scikit-learn định nghĩa nó là trung bình recall trên từng class, dùng cho binary và multiclass để xử lý dữ liệu mất cân bằng. ([scikit-learn.org][6])

---

## Ưu tiên 5 — Thêm “Actual Stacking” bằng chronological split hoặc walk-forward OOF

Nếu bạn muốn đúng đề tài nhất, thêm Stage 4 như sau:

```text
For each walk-forward window:

Train block
  ├── train_base_part
  └── train_meta_part

Base models fit on train_base_part:
  - LightGBM static
  - GRU-only

Base models predict_proba on train_meta_part:
  - lgbm_p_short, lgbm_p_hold, lgbm_p_long
  - gru_p_short, gru_p_hold, gru_p_long

Meta-model fit on these probabilities:
  - Logistic Regression hoặc LightGBM nhỏ

Test block:
  - base models predict_proba
  - meta-model predict final class
```

Meta features nên cực kỳ đơn giản:

```text
lgbm_proba_short
lgbm_proba_hold
lgbm_proba_long
gru_proba_short
gru_proba_hold
gru_proba_long
max_proba_lgbm
max_proba_gru
disagreement_flag
```

Không cần quá nhiều. Dễ giải thích là thắng.

Nếu bạn vẫn muốn giữ “GRU hidden states + LightGBM”, hãy gọi nó là:

```text
Hybrid Fusion baseline
```

Còn mô hình chính:

```text
Hybrid Stacking final
```

Bảng so sánh nên là:

| Model               | Vai trò                         |
| ------------------- | ------------------------------- |
| Majority baseline   | đoán lớp phổ biến               |
| Naive direction     | baseline trading đơn giản       |
| LightGBM static     | tabular ML baseline             |
| GRU-only            | sequence model baseline         |
| Hybrid Fusion       | GRU embedding + static features |
| **Hybrid Stacking** | base proba → meta-model         |

---

## Ưu tiên 6 — Giảm rủi ro overfit của LightGBM thêm chút nữa

Config hiện tại:

```toml
num_leaves = 31
max_depth = 6
learning_rate = 0.02
n_estimators = 300
min_child_samples = 50
feature_fraction = 0.70
reg_lambda = 5.0
```

Khá ổn. Nhưng với tài chính H1, tôi sẽ làm conservative hơn:

```toml
num_leaves = 15
max_depth = 4
min_child_samples = 80
reg_lambda = 10.0
reg_alpha = 0.10
```

LightGBM documentation cũng cảnh báo rằng `num_leaves` lớn có thể gây overfitting, và các tham số như `min_data_in_leaf`, `bagging_fraction`, `feature_fraction` có thể dùng để giảm overfitting. ([LightGBM][7])

Với đồ án, model “ít overfit, kết quả vừa phải nhưng ổn định” tốt hơn model “một window rất đẹp, window sau sập”.

---

## Ưu tiên 7 — Backtest: giữ demo, nhưng bỏ các yếu tố dễ gây tranh cãi

Config backtest hiện tại đã có spread, slippage, commission, confidence threshold, cooldown, max drawdown cutoff, daily loss limit. ([GitHub][3]) Đây là tốt về mặt thực tế, nhưng hơi nhiều nếu chỉ là application demo.

Tôi khuyên report backtest chỉ gồm 2 chế độ:

```text
Demo A: trade all non-Hold signals, fixed lot
Demo B: confidence_threshold = 0.50, fixed lot
```

Không nên nhấn mạnh:

```text
max_drawdown_cutoff
daily_loss_limit
dynamic lot scaling
high confidence position sizing
```

Lý do: các phần này biến đồ án thành trading system optimization. Bạn đã định vị backtest là optional/application demo trong README và architecture docs, nên cứ giữ nó nhẹ. ([GitHub][1])

---

## Ưu tiên 8 — SHAP chỉ dùng cho LightGBM/static hoặc meta-model, đừng cố SHAP toàn GRU

Repo đã ghi hướng phát triển có Feature Importance/SHAP. ([GitHub][1]) Đây là điểm tốt để bảo vệ.

Nhưng nên dùng SHAP đúng chỗ:

```text
SHAP cho LightGBM static
SHAP cho meta-model nếu là tree model
Không cần giải thích sâu hidden states của GRU
```

SHAP documentation mô tả SHAP là phương pháp giải thích output của model theo game theory, còn TreeExplainer dùng Tree SHAP để giải thích các model cây/ensemble cây. ([SHAP][8])

Với hội đồng, biểu đồ nên có:

```text
Top 10 features ảnh hưởng đến LightGBM static
Top meta features:
- lgbm_proba_long
- gru_proba_long
- lgbm_proba_short
- disagreement_flag
```

Cực dễ bảo vệ.

---

## Ưu tiên 9 — Tách “thesis mode” và “experiment mode”

Hiện config có nhiều tham số nâng cao: contrastive pretrain, temperature scaling, PCA, distribution-shift weights, dashboard, ECharts, Streamlit. Những thứ này có ích, nhưng làm bạn dễ loạn.

Nên tách:

```text
configs/thesis.toml
configs/experiment_gru.toml
configs/backtest_aggressive.toml
```

Trong README, chỉ hướng dẫn:

```bash
pixi run workflow --config configs/thesis.toml
```

`thesis.toml` nên đơn giản:

```toml
[model]
architecture = "stacking"
meta_model = "logistic_regression"

[gru]
hidden_size = 32
num_layers = 1
sequence_length = 24
epochs = 30

[validation]
method = "sliding"
train_window_bars = 6240
test_window_bars = 1040
purge_bars = 48
embargo_bars = 50
```

---

## Ưu tiên 10 — Chuẩn hóa câu chuyện học thuật

Bạn nên viết luận văn theo câu chuyện này:

```text
Bài toán:
Dự báo tín hiệu giao dịch CFD vàng H1 thành 3 lớp Short/Hold/Long.

Vấn đề:
Dữ liệu tài chính nhiễu, không IID, dễ leakage, label drift.

Phương pháp:
Triple Barrier labeling + walk-forward validation + purge/embargo.

Mô hình:
Hybrid Stacking kết hợp:
- LightGBM học feature kỹ thuật dạng bảng
- GRU học thông tin chuỗi
- Meta-model kết hợp xác suất dự báo

Đánh giá:
Macro F1, Balanced Accuracy, Precision/Recall từng lớp, Confusion Matrix, so sánh baseline.

Ứng dụng:
Backtest minh họa tín hiệu, không claim lợi nhuận thực tế.
```

Đây là khung bảo vệ sạch nhất.

## Danh sách việc cần làm theo thứ tự

| Ưu tiên | Việc cần làm                                                                      | Tác động                        |
| ------: | --------------------------------------------------------------------------------- | ------------------------------- |
|       1 | Đổi final model thành actual Hybrid Stacking: LGBM proba + GRU proba → meta-model | Khớp tên đề tài                 |
|       2 | Đồng bộ README/docs/config: data range, window size, GRU size, PCA dims, features | Tránh bị bắt lỗi                |
|       3 | Thêm bảng label distribution theo từng walk-forward window                        | Tăng tính học thuật             |
|       4 | Thêm Balanced Accuracy vào report                                                 | Đánh giá tốt hơn khi lệch class |
|       5 | Giảm LightGBM complexity: num_leaves 15, max_depth 4, min_child_samples 80        | Giảm overfit                    |
|       6 | Tách config `thesis.toml` khỏi config thử nghiệm                                  | Đỡ rối, dễ chạy lại             |
|       7 | Dùng SHAP cho LightGBM/static và meta-model                                       | Dễ giải thích                   |
|       8 | Backtest chỉ giữ 2 demo đơn giản                                                  | Không lệch scope                |
|       9 | Thêm bảng model comparison cuối report                                            | Dễ đưa vào luận văn             |
|      10 | Viết sẵn phần “limitations”                                                       | Bảo vệ tốt nếu kết quả xấu      |

## Chốt

Dự án đã đi đúng hướng hơn trước rất nhiều. Nhưng để đúng với đề tài **“Hybrid Stacking”**, bạn nên sửa phần mô hình chính thành:

```text
LightGBM static predict_proba
+ GRU-only predict_proba
→ meta-model
→ Short/Hold/Long
```

Còn kiến trúc hiện tại:

```text
GRU hidden states + static features → LightGBM
```

nên để là **Hybrid Fusion baseline**, không nên gọi là stacking chính.

Làm được 3 việc này là đồ án sẽ chắc hơn hẳn:

```text
1. Actual stacking model.
2. Đồng bộ tài liệu/config.
3. Report rõ: classification primary, backtest demo.
```

[1]: https://github.com/ultimateBroK/thesis-lgbm/ "GitHub - ultimateBroK/thesis-lgbm: Bachelor's Thesis - Hybrid GRU + LightGBM XAU/USD H1 Trading Signals · GitHub"
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html?utm_source=chatgpt.com "StackingClassifier"
[3]: https://raw.githubusercontent.com/ultimateBroK/thesis-lgbm/main/config.toml "raw.githubusercontent.com"
[4]: https://github.com/ultimateBroK/thesis-lgbm/blob/main/docs/ARCHITECTURE.md "thesis-lgbm/docs/ARCHITECTURE.md at main · ultimateBroK/thesis-lgbm · GitHub"
[5]: https://github.com/ultimateBroK/thesis-lgbm/blob/main/docs/EVALUATION.md "thesis-lgbm/docs/EVALUATION.md at main · ultimateBroK/thesis-lgbm · GitHub"
[6]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html?utm_source=chatgpt.com "balanced_accuracy_score"
[7]: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html?utm_source=chatgpt.com "Parameters Tuning — LightGBM 4.6.0.99 documentation"
[8]: https://shap.readthedocs.io/?utm_source=chatgpt.com "Welcome to the SHAP documentation — SHAP latest ..."
