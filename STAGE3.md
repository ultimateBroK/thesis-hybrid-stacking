Đánh giá thẳng: **Stage 3 hiện đã đúng hướng về thư mục, nhưng chưa đúng tinh thần “small thesis pipeline”**.

Trong `src/thesis/models`, hiện bạn chỉ còn 4 file: `baselines.py`, `evaluate.py`, `stacking.py`, `train.py`. Đây là gọn hơn trước. Nhưng file size vẫn lớn: `train.py` khoảng 422 dòng, `stacking.py` khoảng 683 dòng, `evaluate.py` khoảng 496 dòng. Với đồ án CNTT, phần model training như vậy vẫn hơi nặng và khó bảo vệ nếu hội đồng hỏi sâu vào từng nhánh logic. ([GitHub][1])

---

# 1. Stage 3 nên được định nghĩa lại

Hiện `pipeline.py` gọi Stage 3 bằng:

```python
_run_stage(3, config, "run_models", config.paths.model, train_walk_forward)
```

và mô tả là:

```text
Walk-forward model training & evaluation
```

Điều này đúng. Stage 3 nên giữ đúng vai trò này thôi. Không nên chứa regression, confidence filtering phức tạp, calibration, guardrail trading, feature selection động quá nhiều. ([GitHub][2])

Tôi khuyên định nghĩa Stage 3 thành:

```text
Stage 3 — Model Experiment

Input:
  ml_dataset.parquet

Steps:
  1. Load ML dataset
  2. Select fixed feature columns
  3. Generate walk-forward windows
  4. Train baseline models
  5. Train Hybrid Stacking
  6. Save predictions, metrics, feature importance
```

Không hơn.

---

# 2. Vấn đề chính hiện tại

## Vấn đề 1 — `train.py` đang ôm quá nhiều vai trò

Trong `train.py`, hiện có nhiều nhóm logic: walk-forward windows, feature pipeline, target regression, walk-forward loop, dispatch. Chính docstring cũng nói file này chứa “windows, feature pipeline, targets, loop, dispatch”. ([GitHub][3])

Đây là đúng chỗ cần refactor.

Theo rule bạn đưa, `train.py` hiện đang trộn:

```text
high-level orchestration
+ window generation
+ feature selection
+ target handling
+ loop control
+ saving hooks
```

Nên cắt `train.py` xuống còn **orchestrator**.

### Mục tiêu cuối

```python
def train_walk_forward(config: Config) -> None:
    dataset = load_model_dataset(config)
    feature_cols = choose_model_features(dataset, config)
    windows = build_walk_forward_windows(dataset, config)

    results = run_model_experiment(
        dataset=dataset,
        feature_cols=feature_cols,
        windows=windows,
        config=config,
    )

    save_model_experiment(results, config)
```

Đọc là hiểu ngay câu chuyện.

---

## Vấn đề 2 — Stage 3 vẫn còn regression target

`train.py` hiện có `compute_regression_target()` để thêm forward-return target khi `config.model.objective == "regression"`. ([GitHub][3])

Nên **xóa khỏi main Stage 3**.

Đề tài của bạn là:

```text
Hybrid Stacking dự báo tín hiệu giao dịch CFD vàng
```

Tức là classification:

```text
Short / Hold / Long
```

Nếu giữ regression objective, hội đồng có thể hỏi:

> Đề tài của em là phân loại tín hiệu hay hồi quy lợi suất?

Không đáng.

### Khuyến nghị

Xóa hoặc đưa sang archive:

```text
compute_regression_target
model.objective == "regression"
regression_target
```

Giữ Stage 3 là **classification-only**.

---

## Vấn đề 3 — Feature pipeline trong Stage 3 đang quá thông minh

`train.py` hiện có `fit_static_feature_pipeline()` dùng:

```text
DropDuplicateFeatures
DropCorrelatedFeatures
RobustScaler
SelectKBest
```

([GitHub][3])

Cái này nghe hay, nhưng với nhánh `hybrid-small` thì hơi ngược mục tiêu. Bạn đã muốn Stage 2 tạo **bộ feature tối giản, cố định, dễ giải thích**. Vậy Stage 3 không nên tiếp tục tự chọn feature phức tạp nữa.

### Tôi khuyên bỏ dynamic feature selection khỏi Stage 3

Nên dùng:

```text
fixed feature list từ constants.py/config
```

Rồi:

```text
Logistic Regression → dùng scaler
Random Forest → raw features
LightGBM → raw features
Stacking → base probabilities
```

Không cần:

```text
DropDuplicateFeatures
DropCorrelatedFeatures
SelectKBest
```

Nếu muốn có feature selection, đưa nó thành **thí nghiệm phụ**, không phải pipeline chính.

---

## Vấn đề 4 — Label prior features nên bỏ

`train.py` hiện có `_add_label_prior_features()` để tạo:

```text
label_prior_long_lag1
label_prior_short_lag1
```

dựa trên label quá khứ đã shift. ([GitHub][3])

Dù bạn xử lý leakage-safe, tôi vẫn khuyên **bỏ khỏi main experiment**.

Lý do:

* Dễ bị hội đồng hỏi: “Sao feature lại sinh từ label?”
* Làm narrative phức tạp.
* Không cần cho một đồ án ML classification sạch.
* Có thể gây cảm giác “lách” dù không leakage.

### Chốt

Bỏ:

```text
_add_label_prior_features
label_prior_long_lag1
label_prior_short_lag1
REGIME_FEATURES auto append nếu chưa cần
```

Stage 3 chỉ dùng feature từ Stage 2.

---

# 3. Cấu trúc `models/` tôi khuyên

Hiện tại:

```text
models/
  baselines.py
  evaluate.py
  stacking.py
  train.py
```

Tôi khuyên đổi thành:

```text
models/
  validation.py
  estimators.py
  stacking.py
  experiment.py
  artifacts.py
  train.py
```

Nghe có vẻ nhiều file hơn, nhưng mỗi file nhỏ hơn và dễ đọc hơn. Đây là **lược complexity**, không phải tăng complexity.

## Vai trò từng file

```text
validation.py
  WalkForwardWindow
  build_walk_forward_windows

estimators.py
  build_logistic_regression
  build_random_forest
  build_lightgbm
  build_baseline_predictions

stacking.py
  HybridStackingClassifier

experiment.py
  train_one_window
  run_model_experiment

artifacts.py
  save_predictions
  save_metrics
  save_feature_importance
  save_training_history

train.py
  train_walk_forward
```

Nếu muốn ít file hơn:

```text
models/
  validation.py
  estimators.py
  stacking.py
  evaluate.py
  train.py
```

Nhưng **không nên để `train.py` chứa tất cả**.

---

# 4. Stage 3 flow tối giản nên là thế này

```text
ml_dataset.parquet
→ choose fixed features
→ build walk-forward windows
→ for each window:
    train LR
    train RF
    train LightGBM
    train Hybrid Stacking
    predict test window
→ aggregate OOF predictions
→ compute metrics
→ save artifacts
```

Artifacts:

```text
results/<session>/
  predictions/
    oof_predictions.csv

  reports/
    model_metrics.json
    model_comparison.csv
    feature_importance.json
    walk_forward_history.json

  models/
    final_model.joblib
```

Không cần nhiều hơn.

---

# 5. Những thứ nên giữ

## Giữ walk-forward validation

Cái này là điểm học thuật rất tốt. `train.py` hiện đã có `WalkForwardWindow` và `generate_windows()` với purge/embargo/event purge. ([GitHub][3])

Nên giữ, nhưng tách sang `validation.py`.

## Giữ Hybrid Stacking

Đây là lõi đề tài. Nhưng `stacking.py` hiện 683 dòng, quá dài cho “simple thesis implementation”. ([GitHub][4])

Nên rút xuống khoảng **150–250 dòng**.

## Giữ model comparison

Nhưng chỉ giữ 5 model:

```text
Majority Baseline
Logistic Regression
Random Forest
LightGBM
Hybrid Stacking
```

`baselines.py` hiện có cả naive direction, always long, always short, always hold, majority, random. ([GitHub][5]) Với report chính, quá nhiều.

---

# 6. Những thứ nên bỏ khỏi Stage 3 chính

| Thành phần                         |      Khuyến nghị | Lý do                                |
| ---------------------------------- | ---------------: | ------------------------------------ |
| Regression objective               |               Bỏ | Lệch đề tài classification           |
| Label prior features               |               Bỏ | Dễ bị hỏi leakage/narrative phức tạp |
| SelectKBest trong train            |               Bỏ | Feature set đã tối giản ở Stage 2    |
| DropCorrelatedFeatures trong train |               Bỏ | Làm kết quả khó tái hiện/giải thích  |
| Random baseline                    |     Bỏ khỏi main | Ít giá trị bảo vệ                    |
| Always Long/Short/Hold             |     Bỏ khỏi main | Dễ kéo sang trading strategy         |
| Confidence threshold prediction    |     Bỏ khỏi main | Dễ kéo sang calibration              |
| Confidence bins                    | Bỏ hoặc appendix | Không cần cho thesis core            |
| Per-window warning nhiều           |             Giảm | Log nhiễu                            |
| OOF ensemble option                | Bỏ nếu chưa dùng | Tăng nhánh code                      |
| Save architecture copy nhiều file  |               Bỏ | Không cần                            |

---

# 7. `stacking.py` nên đơn giản hóa

Hiện `stacking.py` là file lớn nhất trong models. ([GitHub][4]) Tôi khuyên chốt cứng kiến trúc:

```text
Base:
  Logistic Regression
  Random Forest
  LightGBM

Meta:
  Logistic Regression

Split:
  80% base-train
  20% meta-train
```

Không cần option nhiều.

## API nên như này

```python
class HybridStackingClassifier:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridStackingClassifier":
        base_X, meta_X, base_y, meta_y = chronological_meta_split(X, y)
        self.base_models = fit_base_models(base_X, base_y)
        meta_features = predict_base_probabilities(self.base_models, meta_X)
        self.meta_model = fit_meta_model(meta_features, meta_y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_features = predict_base_probabilities(self.base_models, X)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return class_labels_from_proba(self.predict_proba(X), self.classes_)
```

Các helper đứng phía trên class:

```text
chronological_meta_split
fit_base_models
predict_base_probabilities
fit_meta_model
class_labels_from_proba
```

Không cần comment section. Tên function kể chuyện.

---

# 8. `train.py` nên chỉ còn khoảng 80–120 dòng

## Đề xuất `train.py`

```python
from thesis.models.artifacts import save_model_experiment
from thesis.models.experiment import run_model_experiment
from thesis.models.validation import build_walk_forward_windows


def load_model_dataset(config: Config) -> pl.DataFrame:
    path = Path(config.paths.ml_dataset)
    if not path.exists():
        raise FileNotFoundError(f"ML dataset not found: {path}")
    return pl.read_parquet(path)


def choose_model_features(df: pl.DataFrame, config: Config) -> list[str]:
    features = [c for c in config.features.static_feature_cols if c in df.columns]
    if not features:
        raise ValueError("No configured model features found in ML dataset")
    return features


def train_walk_forward(config: Config) -> None:
    dataset = load_model_dataset(config)
    feature_cols = choose_model_features(dataset, config)
    windows = build_walk_forward_windows(dataset, config)

    experiment = run_model_experiment(
        dataset=dataset,
        feature_cols=feature_cols,
        windows=windows,
        config=config,
    )

    save_model_experiment(experiment, config)
```

Đây là đúng rule bạn đưa:

```text
load → choose → build → run → save
```

Không trộn high-level với low-level.

---

# 9. `validation.py` nên lấy từ `train.py`

Di chuyển:

```text
WalkForwardWindow
generate_windows
_apply_purge_embargo
_apply_event_purge
log_windows
```

sang:

```text
models/validation.py
```

Nhưng đổi tên rõ hơn:

```python
def build_walk_forward_windows(df: pl.DataFrame, config: Config) -> list[WalkForwardWindow]:
    event_end = read_event_end_if_available(df)
    return generate_walk_forward_windows(
        total_bars=len(df),
        train_window_bars=config.validation.train_window_bars,
        test_window_bars=config.validation.test_window_bars,
        step_bars=config.validation.step_bars,
        purge_bars=config.validation.purge_bars,
        embargo_bars=config.validation.embargo_bars,
        min_train_bars=config.validation.min_train_bars,
        event_end=event_end,
    )
```

Nên giữ comment “why” duy nhất:

```python
# Event-based purge is preferred because triple-barrier labels may extend beyond a fixed horizon.
```

Đây là comment tốt. Nó giải thích vì sao không xóa bừa.

---

# 10. `evaluate.py` nên cắt mạnh

`evaluate.py` hiện vừa có prediction helpers, diagnostics, artifact persistence; docstring nói rõ nó được merge từ nhiều module legacy. ([GitHub][6]) Đây là code smell.

Tôi khuyên tách hoặc cắt.

## Giữ trong `evaluate.py`

```text
classification_metrics
confusion_matrix
per_class_metrics
model_comparison_table
```

## Chuyển sang `artifacts.py`

```text
save_oof_predictions
save_training_history
save_feature_importance
save_walk_forward_history
```

## Bỏ khỏi main

```text
confidence_bin
high_conf_70_pct
prediction_entropy
mean_sample_entropy
long/short ratio guardrails
warning nhiều về no LONG/no SHORT
```

Những thứ này hữu ích cho debug, nhưng không cần trong thesis core.

---

# 11. Baseline nên tối giản

Hiện `baselines.py` chạy nhiều baseline: naive direction, always long, always short, always hold, majority class, random. ([GitHub][5])

Tôi khuyên Stage 3 chỉ giữ:

```python
def majority_baseline(y_true: np.ndarray) -> np.ndarray:
    ...

def evaluate_majority_baseline(y_true: np.ndarray) -> dict[str, float]:
    ...
```

Còn model comparison thì gồm:

```text
Majority Baseline
Logistic Regression
Random Forest
LightGBM
Hybrid Stacking
```

Nếu cần `Random Baseline`, đưa vào appendix hoặc test.

---

# 12. Model training nên chạy thế nào?

## Hiện tại nên dùng 4 model

```text
1. Logistic Regression
2. Random Forest
3. LightGBM
4. Hybrid Stacking
```

Đừng thêm XGBoost, SVM, GRU nữa.

## Model factory

`estimators.py`:

```python
def build_logistic_regression(config: Config) -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(...)),
    ])


def build_random_forest(config: Config) -> RandomForestClassifier:
    return RandomForestClassifier(...)


def build_lightgbm(config: Config) -> LGBMClassifier:
    return LGBMClassifier(...)


def build_base_models(config: Config) -> dict[str, Any]:
    return {
        "logistic_regression": build_logistic_regression(config),
        "random_forest": build_random_forest(config),
        "lightgbm": build_lightgbm(config),
    }
```

Không cần feature pipeline ngoài nữa.

---

# 13. `experiment.py` nên kể câu chuyện chính

```python
def slice_window_dataset(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    window: WalkForwardWindow,
) -> WindowDataset:
    ...


def train_models_for_window(
    data: WindowDataset,
    config: Config,
) -> dict[str, TrainedModel]:
    ...


def predict_models_for_window(
    models: dict[str, Any],
    data: WindowDataset,
) -> list[WindowPrediction]:
    ...


def run_one_window(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    window: WalkForwardWindow,
    config: Config,
) -> WindowResult:
    window_data = slice_window_dataset(dataset, feature_cols, window)
    models = train_models_for_window(window_data, config)
    predictions = predict_models_for_window(models, window_data)
    return evaluate_window_predictions(predictions, window_data)


def run_model_experiment(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    windows: list[WalkForwardWindow],
    config: Config,
) -> ModelExperiment:
    window_results = [
        run_one_window(dataset, feature_cols, window, config)
        for window in windows
    ]
    return combine_window_results(window_results)
```

Mỗi function làm đúng một việc. Không cần comment section.

---

# 14. Dataclass nên thêm để code rõ hơn

Thay vì truyền dict nhiều nơi, dùng dataclass:

```python
@dataclass(frozen=True)
class WindowDataset:
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    test_timestamps: np.ndarray


@dataclass(frozen=True)
class ModelPrediction:
    model_name: str
    pred_label: np.ndarray
    pred_proba: np.ndarray | None


@dataclass(frozen=True)
class WindowResult:
    window_index: int
    predictions: list[ModelPrediction]
    metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ModelExperiment:
    feature_cols: list[str]
    window_results: list[WindowResult]
    model_comparison: dict[str, dict[str, float]]
```

Dataclass giúp code tự giải thích hơn comments.

---

# 15. Quy tắc comment cho Stage 3

Nên giữ comment kiểu:

```python
# The meta split must be chronological to avoid training the meta learner
# on base predictions produced from data the base models already fitted.
```

Nên bỏ comment kiểu:

```python
# Fit model
# Predict
# Save results
# Build features
```

Đổi thành function name:

```python
model = fit_hybrid_stacking(data.train_X, data.train_y, config)
predictions = predict_test_window(model, data.test_X)
save_model_predictions(predictions, config)
```

---

# 16. Thứ tự function trong file

Bạn yêu cầu:

> Highest-level function at the bottom, every definition above its caller, related functions grouped by purpose.

Áp vào `train.py`:

```python
def load_model_dataset(...): ...
def choose_model_features(...): ...
def build_walk_forward_windows(...): ...
def run_model_experiment(...): ...
def save_model_experiment(...): ...

def train_walk_forward(...): ...
```

Với `experiment.py`:

```python
# data slicing
def slice_train_frame(...): ...
def slice_test_frame(...): ...
def slice_window_dataset(...): ...

# model fitting
def fit_single_model(...): ...
def fit_models_for_window(...): ...

# prediction
def predict_single_model(...): ...
def predict_models_for_window(...): ...

# window experiment
def run_one_window(...): ...

# full experiment
def run_model_experiment(...): ...
```

---

# 17. P0 changes tôi khuyên làm ngay

| Việc                                     | Lý do                               |
| ---------------------------------------- | ----------------------------------- |
| Xóa regression objective khỏi Stage 3    | Giữ đúng đề tài classification      |
| Xóa label prior features                 | Tránh rủi ro narrative/leakage      |
| Bỏ dynamic feature selection trong train | Stage 2 đã chịu trách nhiệm feature |
| Tách `validation.py` khỏi `train.py`     | Làm `train.py` kể chuyện rõ         |
| Rút `evaluate.py` về metrics core        | Bớt legacy/debug code               |
| Giảm baseline còn Majority               | Report đỡ loãng                     |
| Chốt 4 model: LR, RF, LGBM, Stacking     | Đúng thesis                         |

---

# 18. P1 changes sau đó

| Việc                                           | Lý do                               |
| ---------------------------------------------- | ----------------------------------- |
| Tách `artifacts.py`                            | Save logic không nằm trong evaluate |
| Rút `stacking.py` còn 150–250 dòng             | Lõi đề tài dễ đọc                   |
| Thêm dataclass `WindowDataset`, `WindowResult` | Giảm dict mơ hồ                     |
| Lưu `model_comparison.csv` duy nhất            | Report/dashboard dùng lại           |
| Lưu `oof_predictions.csv` canonical            | Backtest/demo/report đọc chung      |

---

# 19. Stage 3 sau khi refactor nên còn thế này

```text
models/
  validation.py     ~100 dòng
  estimators.py     ~120 dòng
  stacking.py       ~220 dòng
  evaluate.py       ~120 dòng
  artifacts.py      ~100 dòng
  experiment.py     ~180 dòng
  train.py          ~60 dòng
```

Tổng có thể tương đương hoặc ít hơn hiện tại, nhưng quan trọng hơn là **mỗi file một trách nhiệm**. `train.py` không còn là “everything file”.

Nếu muốn ít file hơn:

```text
models/
  validation.py
  estimators.py
  stacking.py
  evaluate.py
  train.py
```

Nhưng vẫn phải bỏ regression, label prior, dynamic selection và confidence diagnostics khỏi main path.

---

# 20. Kết luận

Stage 3 nên trở thành:

```text
ML experiment runner
```

Không phải:

```text
training framework + feature selector + diagnostics engine + artifact manager + regression/classification dispatcher
```

Chốt hướng sửa:

```text
Classification-only.
Fixed feature set.
Walk-forward validation.
4 models.
Simple metrics.
Simple artifacts.
```

Đây là phiên bản dễ bảo vệ nhất, ít code thừa nhất, và đúng với đề tài **Hybrid Stacking dự báo tín hiệu giao dịch CFD vàng**.

[1]: https://github.com/ultimateBroK/thesis-hybrid/tree/hybrid-small/src/thesis/models "thesis-hybrid/src/thesis/models at hybrid-small · ultimateBroK/thesis-hybrid · GitHub"
[2]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-small/src/thesis/pipeline.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-small/src/thesis/models/train.py "raw.githubusercontent.com"
[4]: https://github.com/ultimateBroK/thesis-hybrid/blob/hybrid-small/src/thesis/models/stacking.py "thesis-hybrid/src/thesis/models/stacking.py at hybrid-small · ultimateBroK/thesis-hybrid · GitHub"
[5]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-small/src/thesis/models/baselines.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-small/src/thesis/models/evaluate.py "raw.githubusercontent.com"
