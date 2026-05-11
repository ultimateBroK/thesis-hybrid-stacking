# REFACTOR — hiện trạng `src/thesis` và hướng sửa ít vất vả

Cập nhật: 2026-05-11
Phạm vi: dự án XAU/USD H1, dự báo tín hiệu CFD vàng Long / Hold / Short bằng pipeline học máy có kiểm soát.

## 0. Todo đã xử lý

- [x] Đọc `graphify-out/GRAPH_REPORT.md` trước khi đọc source.
- [x] Đọc các stage chính trong `src/thesis` để xác định ràng buộc thật của code hiện tại.
- [x] Web search xác nhận hướng học thuật: LightGBM, stacking, triple-barrier, purge/embargo, SHAP.
- [x] Viết lại file này theo codebase hiện tại thay vì theo kiến trúc GRU cũ.
- [x] Chạy Stage 3/4/5/6 smoke test sau khi đổi code/config.
- [x] Thử `horizon_bars = 48`; kết quả Hold còn ~1.5%, kém hơn horizon 24 nên đã rollback.
- [x] Giảm feature whitelist theo importance: bỏ `regime_strength`, `upper_wick_ratio`, `lower_wick_ratio`, `volume_zscore_20`.

## 1. Kết luận ngắn

Hướng hợp lý nhất lúc này là giữ bản chính theo:

```text
Classic Hybrid Stacking
= Logistic Regression + Random Forest + LightGBM
  -> meta-model Logistic Regression
  -> tín hiệu Short / Hold / Long
```

Không nên quay lại GRU làm runtime chính. Code hiện tại trong `src/thesis` đã đi theo hướng này rồi:

- `config.toml`: `model.architecture = "stacking"`, `objective = "multiclass"`.
- `src/thesis/stage_4_training/walk_forward/dispatcher.py`: chỉ hỗ trợ `stacking` và `lgbm`.
- `src/thesis/stage_4_training/walk_forward/stacking.py`: đã có stacking time-safe bằng base/meta split theo thời gian.
- `src/thesis/pipeline.py`: mô tả stage 4 là “walk-forward LightGBM or Hybrid Stacking”.

Vì vậy hướng sửa đỡ vất vả nhất không phải là thiết kế lại mô hình, mà là:

1. giữ Classic Stacking làm bản chính;
2. sửa tài liệu/báo cáo cho khớp code hiện tại;
3. chỉ chỉnh label/feature/model theo thứ tự phụ thuộc stage;
4. chạy smoke test theo stage để tránh sửa lan man.

## 2. Xác nhận từ web search

Các điểm dưới đây ủng hộ hướng hiện tại:

1. LightGBM phù hợp làm base learner chính cho dữ liệu tabular nhiều feature. Bài LightGBM gốc mô tả GBDT hiệu quả, huấn luyện nhanh hơn GBDT truyền thống đáng kể trong khi giữ độ chính xác gần tương đương.
2. Stacked generalization dùng dự báo của các mô hình tầng dưới làm input cho learner tầng trên. Điều này khớp với `stacking.py`, nơi xác suất của logreg/rf/lgbm được ghép thành meta-features.
3. Triple-barrier labeling là hướng hợp lý hơn nhãn “giá tăng/giảm sau N nến”, vì mỗi mẫu có take-profit, stop-loss và horizon/expiration.
4. Purging và embargo cần thiết với dữ liệu tài chính vì label có lookahead theo event horizon; nếu train/test overlap theo thông tin tương lai thì đánh giá bị leakage.
5. SHAP là hướng giải thích mô hình phù hợp vì SHAP gán importance cho từng feature ở từng prediction, hữu ích khi bảo vệ câu hỏi “mô hình dựa vào gì?”.

Nguồn đã kiểm tra:

- LightGBM paper: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- Stacked Generalization/Wolpert: https://www.scirp.org/reference/referencespapers?referenceid=2329136
- Triple-barrier overview: https://reasonabledeviations.com/notes/adv_fin_ml/
- Purged CV/leakage: https://en.wikipedia.org/wiki/Purged_cross-validation
- SHAP paper: https://arxiv.org/abs/1705.07874

## 3. Ràng buộc giữa các stage trong code hiện tại

Pipeline chính nằm ở `src/thesis/pipeline.py` và chạy 6 stage theo thứ tự cứng:

```text
Stage 1: Data Preparation
  -> tạo OHLCV parquet
  -> output chính: config.paths.ohlcv

Stage 2: Feature Engineering
  -> đọc OHLCV
  -> tạo feature causal, ATR helper, static model features
  -> output chính: config.paths.features

Stage 3: Label Generation
  -> đọc features + OHLCV nếu cần
  -> dùng ATR từ Stage 2 để tạo triple-barrier labels
  -> output chính: config.paths.labels

Stage 4: Model Training
  -> đọc labels
  -> tạo walk-forward windows bằng event_end/purge/embargo
  -> train stacking hoặc LightGBM
  -> output chính: predictions, model, training history, model comparison

Stage 5: Backtest
  -> đọc predictions/model signals
  -> mô phỏng tín hiệu như application demo
  -> output chính: backtest_results

Stage 6: Reporting
  -> đọc predictions + backtest + feature importance
  -> tạo thesis report, model_metrics, model_evaluation, model_comparison
```

### Ràng buộc quan trọng nhất

#### Stage 2 phụ thuộc Stage 1

File: `src/thesis/stage_2_features/engineering.py`

- Bắt buộc có `config.paths.ohlcv`.
- OHLCV phải có timestamp tăng dần, không duplicate.
- Feature output được giới hạn bởi `build_feature_output_cols(config)`.
- Feature list thật lấy từ `config.features.static_feature_cols`, mặc định là `CORE_STATIC_FEATURES` trong `src/thesis/shared/constants.py`.

Hướng sửa an toàn:

- Nếu thêm/xóa feature model-facing, sửa `CORE_STATIC_FEATURES` hoặc `config.features.static_feature_cols`.
- Nếu thêm indicator mới, phải gọi nó trong `generate_features()`, không chỉ viết function trong `_indicators/`.
- Không đưa raw `open/high/low/close/volume`, `timestamp`, `label`, barrier metadata vào feature model.

#### Stage 3 phụ thuộc Stage 2

File: `src/thesis/stage_3_labels/labeling.py`

- Bắt buộc Stage 2 tạo được `atr_{config.features.atr_period}`.
- Label dùng:
  - `config.labels.atr_tp_multiplier`
  - `config.labels.atr_sl_multiplier`
  - `config.labels.horizon_bars`
  - `config.labels.min_atr`
- Output label là `+1` Long, `0` Hold, `-1` Short, `-2` censored.
- Rows censored bị drop trước training.
- Có `event_end` và `sample_weight` để Stage 4 dùng event-time purge và average uniqueness.

Hướng sửa an toàn:

- Nếu chỉnh `horizon_bars` hoặc ATR multipliers, phải rerun Stage 3 trở đi.
- Nếu muốn backtest đo cùng event với label, phải giữ backtest TP/SL khớp label TP/SL.
- Đừng đổi encoding label nếu chưa sửa toàn bộ metric/report/backtest, vì nhiều nơi đang giả định class order `[-1, 0, 1]`.

#### Stage 4 phụ thuộc Stage 3

File chính:

- `src/thesis/stage_4_training/walk_forward/dispatcher.py`
- `src/thesis/stage_4_training/walk_forward/stacking.py`
- `src/thesis/stage_4_training/walk_forward/lgbm.py`
- `src/thesis/stage_4_training/validation.py`

Ràng buộc:

- `dispatcher.py` chỉ nhận `architecture in {"stacking", "lgbm"}`.
- `stacking.py` hiện chỉ nhận `objective = "multiclass"`.
- Walk-forward dùng `event_end` nếu labels có cột này; nếu không thì fallback fixed purge.
- Base/meta split trong stacking là chronological split: đầu train window để train base models, đuôi train window để train meta-model.
- Meta-feature là xác suất Short/Hold/Long từ từng base learner.

Hướng sửa an toàn:

- Không sửa GRU nữa nếu mục tiêu là hoàn thành đồ án.
- Nếu sửa mô hình, sửa trong `stacking.py` hoặc `lgbm.py`, không động vào toàn pipeline.
- Nếu thêm base model mới, cần thêm vào `_BASE_MODEL_ALIASES`, `_build_sklearn_base_model()`, config `stacking_base_models`, và kiểm tra `_stack_probability_features()` vẫn nhận ma trận 3 cột.
- Nếu đổi sang binary, đây là refactor lớn: phải sửa label config/schema, `_CLASS_ORDER`, metric/report/backtest. Không nên làm nếu đang cần ổn định.

#### Stage 5 phụ thuộc Stage 3 và Stage 4

File: `src/thesis/pipeline.py`, `src/thesis/stage_5_backtest/`

Ràng buộc cứng trong `_run_backtest_with_barrier_guard()`:

```text
labels.tp == backtest.tp
labels.sl == backtest.sl
```

Nếu không khớp, pipeline raise `ValueError`.

Ý nghĩa: label học một kiểu TP/SL thì backtest cũng phải dùng cùng kiểu thoát lệnh, tránh việc training target và execution target đo hai bài toán khác nhau.

Hướng sửa an toàn:

- Khi đổi `[labels] atr_tp_multiplier` hoặc `atr_sl_multiplier`, đổi luôn `[backtest] atr_tp_multiplier` và `atr_stop_multiplier`.
- Backtest chỉ nên giữ vai trò minh họa ứng dụng, không phải bằng chứng chính của luận văn.

#### Stage 6 phụ thuộc Stage 4 và Stage 5

File: `src/thesis/stage_6_reporting/generation.py`

Ràng buộc:

- Đọc predictions từ `config.paths.predictions`.
- Đọc backtest từ `config.paths.backtest_results` nếu tồn tại.
- Ghi `model_metrics.json`, `model_evaluation.md`, `model_comparison.csv/md`.
- Classification metrics là primary evidence; backtest được ghi rõ là application demo.

Hướng sửa an toàn:

- Nếu đổi tên model/architecture, sửa `_model_label()` và model comparison/report labels.
- Nếu đổi số lớp, phải sửa report metrics, confusion matrix, per-class table.
- Nên giữ report nhấn mạnh ML metrics: Accuracy, Directional Accuracy, Macro F1, per-class F1, confusion matrix.

## 4. Hướng sửa file theo mức ưu tiên

### Ưu tiên 1 — Giữ architecture hiện tại, không mở lại GRU

Không cần sửa nhiều code. Chỉ cần đảm bảo các file sau cùng nói một câu chuyện:

- `README.md`
- `docs/*` nếu còn nhắc GRU là runtime chính
- `REFACTOR.md`
- `src/thesis/stage_6_reporting/benchmarks.py` nếu label hiển thị còn cũ
- `src/thesis/stage_6_reporting/sections.py` nếu methodology còn mô tả GRU

Thông điệp thống nhất:

```text
Bản chính là Classic Hybrid Stacking: Logistic Regression + Random Forest + LightGBM,
meta-model Logistic Regression, đánh giá bằng walk-forward với purge/embargo.
GRU chỉ là hướng phát triển/thử nghiệm lịch sử, không phải runtime chính.
```

### Ưu tiên 2 — Kiểm tra label distribution trước khi sửa model

File cần xem/sửa:

- `config.toml`
- `src/thesis/stage_3_labels/labeling.py`

Config hiện tại hợp lý để bắt đầu:

```toml
[labels]
atr_tp_multiplier = 2.0
atr_sl_multiplier = 2.0
horizon_bars = 24
```

Nếu Hold quá nhiều hoặc model khó học:

- thử `horizon_bars = 48` trước;
- giữ TP/SL đối xứng `2.0 / 2.0`;
- không vội đổi binary vì code hiện tại đang tối ưu cho multiclass.

Tiêu chí nhìn nhanh:

```text
Short: 25–40%
Hold : 20–50%
Long : 25–40%
```

Không cần hoàn hảo, nhưng nếu Hold 80–90% hoặc Long/Short lệch cực mạnh thì phải sửa label trước model.

### Ưu tiên 3 — Giảm feature bằng whitelist, không xóa indicator vội

File cần sửa nếu muốn giảm feature:

- `src/thesis/shared/constants.py` (`CORE_STATIC_FEATURES`)
- hoặc override `features.static_feature_cols` qua config nếu muốn ít đụng code

Feature hiện tại đang ở mức vừa phải, gồm trend, momentum, volatility, position, candle, session, volume. Nếu cần bản dễ bảo vệ hơn, ưu tiên giữ:

```text
ema34_vs_ema89
close_vs_ema_34
adx_14
ema_slope_20
return_1h
return_4h
macd_hist_atr
rsi_14
atr_pct_close
atr_ratio
high_low_range_20
price_dist_ratio
price_position_20
pivot_position
vwap
sess_asia
sess_london
sess_ny_am
sess_ny_pm
```

Có thể bỏ trước nếu nhiễu/importance thấp:

```text
candle_body_ratio
upper_wick_ratio
lower_wick_ratio
volume_zscore_20
atr_percentile
regime_strength
```

Nhưng chỉ nên bỏ sau khi xem `feature_importance.json` từ kết quả thật.

### Ưu tiên 4 — Chỉ tinh chỉnh model sau khi label/feature ổn

File cần sửa:

- `config.toml`
- `src/thesis/stage_4_training/walk_forward/stacking.py` nếu thêm/xóa base learner
- `src/thesis/stage_4_training/lgbm/utils.py` nếu chỉnh LightGBM internals

Config hiện tại phù hợp mục tiêu “ít overfit, dễ giải thích”:

```toml
[model]
architecture = "stacking"
objective = "multiclass"
num_leaves = 15
max_depth = 4
learning_rate = 0.03
n_estimators = 300
min_child_samples = 80
feature_fraction = 0.70
reg_lambda = 10.0
stacking_base_models = ["logistic_regression", "random_forest", "lightgbm"]
stacking_meta_model = "logistic_regression"
stacking_meta_fraction = 0.20
```

Không nên tăng complexity trước khi có bảng so sánh:

```text
Naive baseline
Logistic Regression
Random Forest
LightGBM
Hybrid Stacking
```

## 5. Checklist sửa đổi an toàn theo stage

### Khi sửa Stage 2 feature

- [ ] Sửa feature generator hoặc `CORE_STATIC_FEATURES`.
- [ ] Rerun Stage 2 trở đi.
- [ ] Kiểm tra `data/processed/features.feature_list.json`.
- [ ] Đảm bảo không có null sau warm-up.

Command:

```bash
pixi run python main.py --stage 2 --force
```

### Khi sửa Stage 3 label

- [ ] Sửa `[labels]`.
- [ ] Nếu TP/SL đổi, sửa `[backtest]` cho khớp.
- [ ] Rerun Stage 3 trở đi.
- [ ] Kiểm tra phân phối label trong log hoặc labels parquet.

Command:

```bash
pixi run python main.py --stage 3 --force
```

### Khi sửa Stage 4 model

- [ ] Giữ `objective = "multiclass"` nếu dùng `architecture = "stacking"`.
- [ ] Không thêm `multi_class="auto"` vào LogisticRegression vì sklearn mới có thể lỗi.
- [ ] Chạy smoke test Stage 4.
- [ ] Kiểm tra `predictions/final_predictions.parquet`, `reports/model_comparison.*`, `reports/walk_forward_history.json`.

Command:

```bash
pixi run ruff check src
pixi run python -m compileall -q src tests
pixi run python main.py --stage 4 --force
```

### Khi sửa Stage 5 backtest

- [ ] Giữ label/backtest ATR barriers khớp.
- [ ] Không dùng backtest làm metric chính trong báo cáo.
- [ ] Kiểm tra phí, spread, slippage có được ghi rõ.

### Khi sửa Stage 6 report

- [ ] Report phải nói “classification metrics là primary”.
- [ ] Backtest là application demo.
- [ ] Bảng model comparison phải có base models + Hybrid Stacking.
- [ ] Nếu model không thắng LightGBM, viết trung thực là stacking không luôn vượt model đơn lẻ trên dữ liệu nhiễu cao.

## 6. Câu chuyện bảo vệ nên dùng

Đề tài không cần chứng minh chiến lược CFD vàng sinh lời ổn định. Câu chuyện an toàn hơn là:

```text
Đồ án xây dựng pipeline học máy để dự báo tín hiệu giao dịch CFD vàng trên dữ liệu XAU/USD H1.
Trọng tâm là quy trình đánh giá đúng cho time series tài chính: feature causal,
triple-barrier labeling, walk-forward validation, purge/embargo chống leakage,
so sánh baseline và mô hình, giải thích feature importance/SHAP.
Backtest chỉ minh họa cách tín hiệu có thể được chuyển thành giao dịch.
```

Khi hội đồng hỏi “Hybrid ở đâu?”:

```text
Hybrid nằm ở kiến trúc stacking: các base learners có bản chất khác nhau
(Logistic Regression tuyến tính, Random Forest bagging tree, LightGBM boosting tree)
cùng tạo xác suất Short/Hold/Long. Meta-model học cách kết hợp các xác suất này
để tạo tín hiệu cuối cùng. Vì vậy mô hình không phụ thuộc một thuật toán đơn lẻ.
```

Khi Hybrid không hơn LightGBM:

```text
Kết quả cho thấy stacking không luôn vượt LightGBM đơn lẻ trên dữ liệu tài chính nhiễu cao.
Đây là kết quả hợp lý và trung thực: tăng độ phức tạp có thể làm overfit.
Đóng góp chính của đồ án là pipeline đánh giá có kiểm soát, chống leakage,
so sánh minh bạch và phân tích điều kiện mô hình hoạt động tốt hoặc không tốt.
```

## 7. Kế hoạch làm tiếp ít vất vả nhất

1. Không sửa architecture nữa.
2. Chạy Stage 3 để xem label distribution.
3. Nếu label lệch: chỉ thử 2 cấu hình:
   - `horizon_bars = 24`, TP/SL `2.0/2.0`
   - `horizon_bars = 48`, TP/SL `2.0/2.0`
4. Chạy Stage 4 stacking.
5. So sánh `hybrid_stacking` với `lgbm`, `rf`, `logreg` trong model comparison.
6. Nếu kết quả xấu, giảm feature theo importance trước; không thêm mô hình mới.
7. Chạy Stage 5/6 để lấy report hoàn chỉnh.
8. Viết luận văn theo hướng “nghiên cứu so sánh mô hình có kiểm soát”, không theo hướng “chiến lược kiếm tiền”.

## 8. File map nhanh

```text
src/thesis/pipeline.py
  Điều phối 6 stage, cache, barrier guard cho backtest.

src/thesis/shared/config.py
  Dataclass config thật. Nếu config.toml có key sai, loader sẽ báo unknown key.

src/thesis/shared/constants.py
  CORE_STATIC_FEATURES và EXCLUDE_COLS.

src/thesis/shared/feature_registry.py
  Source of truth cho feature output, label output, exclude cols.

src/thesis/stage_2_features/engineering.py
  Orchestrator tạo feature. Thêm indicator thì phải gọi tại đây.

src/thesis/stage_3_labels/labeling.py
  Triple-barrier labels, event_end, average uniqueness sample_weight.

src/thesis/stage_4_training/validation.py
  Walk-forward windows, event-time purge, embargo.

src/thesis/stage_4_training/walk_forward/dispatcher.py
  Chỉ route stacking/lgbm.

src/thesis/stage_4_training/walk_forward/stacking.py
  Classic Hybrid Stacking chính.

src/thesis/stage_4_training/walk_forward/lgbm.py
  LightGBM-only baseline/ablation.

src/thesis/stage_5_backtest/
  Backtest application demo.

src/thesis/stage_6_reporting/generation.py
  Thesis markdown + model metrics + model comparison.
```

## 9. Kết quả thực hiện smoke test 2026-05-11

Phiên kết quả mới nhất:

```text
results/XAUUSD_1H_20260511_231114/
```

Kết quả chính:

```text
Label distribution: Short 43.6%, Hold 9.0%, Long 47.4%
Accuracy: 0.3397
Directional Accuracy: 0.4938
Macro F1: 0.3162
Per-class F1: Short 0.3684, Hold 0.1848, Long 0.3952
Backtest demo: total return 0.04%, max drawdown -3.13%, win rate 47.67%, 172 trades
```

So sánh mô hình:

```text
Hybrid Stacking     acc 0.3397, macro F1 0.3162
Logistic Regression acc 0.3582, macro F1 0.3182
Random Forest       acc 0.3580, macro F1 0.3265
LightGBM            acc 0.3770, macro F1 0.3281
```

Diễn giải nên dùng trong luận văn: Hybrid Stacking không vượt LightGBM ở lần chạy này. Đây là kết quả hợp lệ và cần viết trung thực: dữ liệu tài chính nhiễu cao, stacking tăng độ phức tạp nhưng không luôn cải thiện OOS. Đóng góp chính vẫn là pipeline đánh giá có kiểm soát.

## 10. Quy tắc vàng để đỡ vất vả

- Sửa theo thứ tự: label -> feature -> model -> backtest -> report.
- Mỗi lần chỉ đổi một nhóm config/code.
- Sau khi đổi label, luôn rerun từ Stage 3.
- Sau khi đổi feature, luôn rerun từ Stage 2.
- Không đánh giá model bằng random split.
- Không để backtest là bằng chứng chính.
- Không mở lại GRU nếu mục tiêu là hoàn thành luận văn ổn định.
- Không đổi binary/multiclass nếu chưa sẵn sàng sửa toàn bộ metric/report/backtest.
