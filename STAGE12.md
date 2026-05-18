Đúng hướng. Nhưng nếu xét riêng **Stage 1 + Stage 2**, tôi khuyên bạn chỉnh thêm theo hướng này:

```text
Stage 1: ticks → clean OHLCV + microstructure bars
Stage 2: OHLCV/microstructure → minimal features → labels → ML dataset
```

Hiện tại pipeline của bạn đã có flow rõ:

```python
prepare_dataset(config)
build_features(config)
build_labels(config)
build_ml_dataset(config)
```

Trong `pipeline.py`, Stage 2 đang gọi tuần tự `build_features → build_labels → build_ml_dataset`, đây là flow đúng và nên giữ. ([GitHub][1])

Vấn đề chính không nằm ở flow, mà nằm ở **feature set đang nhiều và chưa tận dụng tốt data raw đang có**.

---

# 1. Stage 1 nên đổi nhẹ: không chỉ OHLCV, mà là OHLCV+

File raw bạn upload có:

```text
timestamp
ask
bid
ask_volume
bid_volume
```

Hiện Stage 1 đang dùng bid/ask để tạo `microprice`, tổng volume, `tick_count`, `avg_spread`. Code `_microprice()` đã tính microprice từ ask/bid và opposing-side volume; `_aggregate_file()` cũng đã xuất `tick_count` và `avg_spread`. ([GitHub][2])

Nhưng sau đó `constants.py` lại exclude:

```python
"volume",
"avg_spread",
"tick_count",
```

khỏi model features. ([GitHub][3])

Tức là bạn **đã có data tốt**, nhưng Stage 2 chưa tận dụng đủ.

## Tôi khuyên Stage 1 output thành OHLCV+

Thay vì chỉ nghĩ là:

```text
timestamp, open, high, low, close, volume, tick_count, avg_spread
```

nên thêm:

```text
bid_volume
ask_volume
volume_imbalance
spread
spread_pct_close
```

### Output Stage 1 đề xuất

```text
timestamp
open
high
low
close
volume
tick_count
avg_spread
bid_volume
ask_volume
volume_imbalance
spread_pct_close
```

Trong đó:

```text
volume = ask_volume + bid_volume
volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + eps)
spread_pct_close = avg_spread / close
```

Đây là phần **tận dụng data đang có** nhất.

Vì dữ liệu của bạn không chỉ có OHLC, mà có **bid/ask + volume hai bên**. Nếu bỏ phí phần này thì hơi đáng tiếc.

---

# 2. Stage 1 nên đơn giản hóa trách nhiệm

Hiện `prepare_dataset.py` đang làm khá nhiều việc:

```text
validate OHLCV
check gap
classify calendar gap
discover files
aggregate ticks
clean/dedupe
filter date range
log quality
save json
persist parquet
```

Về mặt chạy được thì ổn. Nhưng theo luật code bạn đưa, file này đang hơi nhiều tầng high-level + low-level trộn nhau.

## Cấu trúc tôi khuyên

```text
data/
  prepare_dataset.py
  aggregate_ticks.py
  quality.py
```

Nếu muốn ít file hơn thì vẫn có thể giữ 1 file, nhưng chia nhóm function rõ ràng.

### Function order nên như này

Theo rule của bạn: **definition above caller, highest-level function at bottom**.

```python
# low-level price helpers
def valid_quote_filter() -> pl.Expr: ...
def microprice_expr() -> pl.Expr: ...
def volume_imbalance_expr() -> pl.Expr: ...

# one-file aggregation
def read_tick_file(path: Path) -> pl.DataFrame: ...
def clean_ticks(ticks: pl.DataFrame) -> pl.DataFrame: ...
def aggregate_ticks_to_bars(ticks: pl.DataFrame, timeframe: str) -> pl.DataFrame: ...
def keep_file_month_only(bars: pl.DataFrame, file_stem: str) -> pl.DataFrame: ...

# all-file aggregation
def find_tick_files(raw_dir: Path) -> list[Path]: ...
def aggregate_tick_files(files: list[Path], timeframe: str) -> pl.DataFrame: ...

# post-processing
def dedupe_bars(bars: pl.DataFrame) -> pl.DataFrame: ...
def keep_date_range(bars: pl.DataFrame, config: Config) -> pl.DataFrame: ...
def write_data_summary(bars: pl.DataFrame, config: Config) -> None: ...
def write_ohlcv(bars: pl.DataFrame, config: Config) -> None: ...

# public API at bottom
def prepare_dataset(config: Config) -> None:
    files = find_tick_files(Path(config.paths.data_raw))
    bars = aggregate_tick_files(files, config.data.timeframe)
    bars = dedupe_bars(bars)
    bars = keep_date_range(bars, config)
    write_data_summary(bars, config)
    write_ohlcv(bars, config)
```

Nhìn là thấy câu chuyện:

```text
find → aggregate → clean → filter → summarize → save
```

Không cần comment nhiều.

---

# 3. Stage 2 hiện đang quá nhiều feature

Log session của bạn cho thấy Stage 2 đang tạo **32 feature columns**:

```text
adx_14
adx_lag1
asia_range_atr
atr_pct_close
atr_percentile
atr_ratio
bb_pctb
candle_body_ratio
close_sma50_ratio
close_vs_ema_34
day_of_week
ema34_vs_ema89
ema_slope_20
ema_slope_lag1
high_low_range_20
macd_hist_atr
pivot_position
price_dist_ratio
price_position_20
return_1h
return_1h_lag1
return_4h
rsi_14
rsi_lag1
sess_asia
sess_london
sess_ny_am
sess_ny_pm
stoch_k
vol_rsi_interaction
vwap
williams_r
```

Trong khi feature importance mới nhất của bạn chỉ nổi bật khoảng 10–16 feature:

```text
vwap
atr_pct_close
ema34_vs_ema89
adx_14
close_sma50_ratio
day_of_week
high_low_range_20
atr_percentile
bb_pctb
return_4h
asia_range_atr
candle_body_ratio
return_1h_lag1
sess_london
sess_ny_am
sess_ny_pm
```

Điều này nói khá rõ: **feature set nên cắt xuống còn 16–20 feature**.

---

# 4. Bộ feature tối giản tôi khuyên

Tôi khuyên dùng **18 feature**, chia thành 5 nhóm.

## Nhóm A — Return / momentum ngắn hạn

Giữ:

```text
return_1h
return_4h
rsi_14
macd_hist_atr
```

Không cần giữ cả:

```text
stoch_k
williams_r
rsi_lag1
```

Vì RSI, Stochastic, Williams %R đều là oscillator; giữ cả ba dễ dư.

---

## Nhóm B — Trend

Giữ:

```text
ema34_vs_ema89
close_vs_ema_34
adx_14
close_sma50_ratio
```

Có thể bỏ:

```text
ema_slope_20
ema_slope_lag1
adx_lag1
```

Vì trend đã có `ema34_vs_ema89`, `close_vs_ema_34`, `adx_14`.

---

## Nhóm C — Volatility / range

Giữ:

```text
atr_pct_close
atr_percentile
high_low_range_20
bb_pctb
```

Có thể bỏ:

```text
atr_ratio
```

Vì `atr_pct_close` + `atr_percentile` đã đủ mô tả volatility absolute + relative.

---

## Nhóm D — Position / session context

Giữ:

```text
price_position_20
close_vs_vwap_atr
day_of_week
sess_london
sess_ny_am
```

Tôi **khuyên thay `vwap` bằng `close_vs_vwap_atr`**.

Hiện `vwap` là raw price level. Feature importance của nó cao nhất, nhưng raw price level trong time-series tài chính dễ gây nhiễu vì model học theo regime/time period thay vì cấu trúc giá. Nên dùng dạng normalized:

```text
close_vs_vwap_atr = (close - vwap) / atr
```

Tức là không bỏ VWAP, nhưng không feed raw VWAP vào model.

Có thể bỏ:

```text
pivot_position
price_dist_ratio
sess_asia
sess_ny_pm
```

---

## Nhóm E — Microstructure từ data bid/ask

Đây là nhóm nên thêm để tận dụng raw data.

Thêm:

```text
spread_pct_close
tick_count_zscore_20
volume_zscore_20
volume_imbalance
```

Vì data raw của bạn có bid/ask và bid/ask volume. Đây là lợi thế so với dataset OHLCV thường.

---

# 5. Feature set cuối tôi đề xuất

## Bản 18 feature

```python
MINIMAL_STATIC_FEATURES = [
    # Return / momentum
    "return_1h",
    "return_4h",
    "rsi_14",
    "macd_hist_atr",

    # Trend
    "ema34_vs_ema89",
    "close_vs_ema_34",
    "close_sma50_ratio",
    "adx_14",

    # Volatility / range
    "atr_pct_close",
    "atr_percentile",
    "high_low_range_20",
    "bb_pctb",

    # Position / session
    "price_position_20",
    "close_vs_vwap_atr",
    "day_of_week",
    "sess_london",
    "sess_ny_am",

    # Microstructure
    "spread_pct_close",
    "tick_count_zscore_20",
    "volume_imbalance",
]
```

Thực ra đây là 20 feature. Nếu muốn đúng 18, bỏ:

```text
close_sma50_ratio
bb_pctb
```

## Bản tôi khuyên dùng để bảo vệ: 18 feature

```python
MINIMAL_STATIC_FEATURES = [
    "return_1h",
    "return_4h",
    "rsi_14",
    "macd_hist_atr",

    "ema34_vs_ema89",
    "close_vs_ema_34",
    "adx_14",

    "atr_pct_close",
    "atr_percentile",
    "high_low_range_20",

    "price_position_20",
    "close_vs_vwap_atr",

    "day_of_week",
    "sess_london",
    "sess_ny_am",

    "spread_pct_close",
    "tick_count_zscore_20",
    "volume_imbalance",
]
```

Đây là bộ gọn, dễ giải thích, tận dụng đúng dữ liệu đang có.

---

# 6. Những feature nên bỏ khỏi main experiment

Tôi khuyên bỏ khỏi main feature set:

```text
stoch_k
williams_r
rsi_lag1
adx_lag1
ema_slope_lag1
return_1h_lag1
atr_ratio
pivot_position
price_dist_ratio
asia_range_atr
sess_asia
sess_ny_pm
vol_rsi_interaction
vwap
```

Không phải chúng vô dụng, nhưng chúng làm đồ án khó giải thích hơn.

Nếu muốn giữ lại cho “future work” thì được.

---

# 7. Stage 2 nên đổi từ “add tất cả rồi select” sang “build theo recipe”

Hiện `build_features.py` import rất nhiều hàm indicator: ADX, ATR, percentile, Bollinger, calendar, EMA, lagged, MACD, pivot, price action, session, stochastic, Williams %R, volume interaction, VWAP... ([GitHub][4])

Về code smell: function `build_features()` đang là một danh sách dài các bước low-level:

```python
df = add_atr(df, config)
df = add_adx(df, config)
df = add_ema_slope(df, config)
df = add_ema_crossover(df, config)
...
```

Nó chạy được, nhưng đọc hơi mệt. Theo luật code bạn đưa, function này đang vừa “orchestrate” vừa biết quá nhiều chi tiết.

## Tôi khuyên đổi thành

```python
def build_features(config: Config) -> None:
    bars = read_bars(config)
    features = create_minimal_features(bars, config)
    features = keep_model_columns(features, config)
    features = drop_warmup_rows(features, model_feature_cols(config))
    validate_features(features)
    write_features(features, config)
```

Trong đó:

```python
def create_minimal_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    df = add_core_price_features(df, config)
    df = add_trend_features(df, config)
    df = add_volatility_features(df, config)
    df = add_position_features(df, config)
    df = add_microstructure_features(df, config)
    df = add_time_features(df, config)
    return df
```

Mỗi function kể một ý nghĩa nghiệp vụ, không phải liệt kê indicator rải rác.

---

# 8. Stage 2 file structure nên gọn lại

Hiện tại:

```text
dataset/
  _label_numba.py
  build_features.py
  build_labels.py
  build_ml_dataset.py
  indicators.py
```

Tôi khuyên đổi thành:

```text
dataset/
  features.py
  labels.py
  assemble.py
  indicators.py
```

Hoặc nếu muốn giữ tên hiện tại:

```text
dataset/
  build_features.py
  build_labels.py
  build_ml_dataset.py
  feature_blocks.py
  label_engine.py
```

Tôi thích nhất:

```text
dataset/
  build_features.py      # public orchestration
  feature_blocks.py      # add_trend_features, add_volatility_features...
  indicators.py          # low-level formulas only
  build_labels.py
  build_ml_dataset.py
```

`indicators.py` chỉ nên chứa công thức nhỏ:

```text
atr_expr
rsi_expr
ema_expr
macd_expr
rolling_zscore_expr
```

Còn `feature_blocks.py` chứa ý nghĩa:

```text
add_return_features
add_trend_features
add_volatility_features
add_position_features
add_microstructure_features
add_time_features
```

---

# 9. Cách viết function theo rule của bạn

## Không nên

```python
def build_features(config):
    # initialization
    ...
    # trend
    ...
    # momentum
    ...
    # volatility
    ...
    # validate
    ...
```

Nếu cần comment section như vậy, function đang ôm quá nhiều.

## Nên

```python
def build_features(config: Config) -> None:
    bars = read_ohlcv_bars(config)
    features = create_model_features(bars, config)
    features = clean_feature_rows(features, model_feature_cols(config))
    write_features(features, config)
```

Câu chuyện rõ:

```text
read → create → clean → write
```

---

# 10. Comments nên giữ ở đâu?

Nên bỏ comment kiểu:

```python
# -- trend --
# -- momentum --
# -- volatility --
```

Thay bằng function name:

```python
df = add_trend_features(df, config)
df = add_momentum_features(df, config)
df = add_volatility_features(df, config)
```

Nên giữ comment dạng “why”:

```python
# Keep OHLC columns because label construction needs future high/low barriers.
```

hoặc:

```python
# Raw VWAP is intentionally excluded; close_vs_vwap_atr is more stationary.
```

Đây là comment tốt vì bảo vệ code khỏi người khác “sửa nhầm”.

---

# 11. Đề xuất Stage 2 mới

## `build_features.py`

```python
def read_ohlcv_bars(config: Config) -> pl.DataFrame:
    path = Path(config.paths.ohlcv)
    if not path.exists():
        raise FileNotFoundError(f"OHLCV not found: {path}")
    return pl.read_parquet(path)


def create_model_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    df = add_return_features(df, config)
    df = add_trend_features(df, config)
    df = add_volatility_features(df, config)
    df = add_position_features(df, config)
    df = add_microstructure_features(df, config)
    df = add_time_features(df, config)
    return df


def select_feature_output(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    columns = build_feature_output_cols(config)
    return df.select([c for c in columns if c in df.columns])


def write_feature_artifacts(df: pl.DataFrame, config: Config) -> None:
    out_path = Path(config.paths.features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    write_feature_list(out_path, model_feature_cols(config))


def build_features(config: Config) -> None:
    bars = read_ohlcv_bars(config)
    validate_ohlcv_bars(bars, config)

    features = create_model_features(bars, config)
    features = select_feature_output(features, config)
    features = drop_invalid_feature_rows(features, model_feature_cols(config))

    validate_feature_frame(features, config)
    write_feature_artifacts(features, config)
```

Nếu function này vẫn hơi dài thì chấp nhận được. Nó là public orchestration.

---

# 12. `feature_blocks.py` nên như này

```python
def add_return_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    return df.with_columns([
        log_return(1).alias("return_1h"),
        log_return(4).alias("return_4h"),
    ])


def add_trend_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    atr = pl.col(f"atr_{config.features.atr_period}")
    ema34 = ema("close", 34)
    ema89 = ema("close", 89)

    return df.with_columns([
        ((pl.col("close") - ema34) / (atr + FEATURE_EPS)).alias("close_vs_ema_34"),
        ((ema34 - ema89) / (atr + FEATURE_EPS)).alias("ema34_vs_ema89"),
        adx(config.features.adx_period).alias("adx_14"),
    ])
```

Tên function nói rõ nó tạo gì. Không cần comment.

---

# 13. Label stage nên giữ, nhưng giảm phụ trợ

`build_labels.py` hiện dùng triple-barrier, có `compute_average_uniqueness`, `event_end`, `sample_weight`. ([GitHub][5])

Với đồ án hiện tại, nếu bạn đang muốn gọn:

## Nên giữ

```text
label
upper_barrier
lower_barrier
touched_bar
event_end
```

## Có thể bỏ khỏi main

```text
sample_weight
average_uniqueness
```

Nếu Stage 3 không thật sự cần sample weight, bỏ sẽ giảm giải thích.

Với thesis, triple-barrier đã đủ học thuật. Average uniqueness là kiến thức hay nhưng có thể làm hội đồng hỏi sâu.

---

# 14. Build ML dataset nên rất đơn giản

`build_ml_dataset.py` chỉ nên làm:

```text
load labels
select timestamp + features + label
validate no null
save ml_dataset.parquet
save label_distribution.json
save feature_list.json
```

Không nên có logic chọn feature quá phức tạp ở đây. Feature selection nên nằm ở config/constants.

---

# 15. Đề xuất pipeline thống nhất

## Hiện tại

```text
Stage 1: prepare_dataset
Stage 2: build_features → build_labels → build_ml_dataset
```

## Nên đổi wording

```text
Stage 1: Market Data Preparation
  raw tick parquet
  → clean tick quotes
  → hourly OHLCV+ bars
  → data_summary.json

Stage 2: ML Dataset Construction
  OHLCV+ bars
  → minimal causal features
  → triple-barrier labels
  → ml_dataset.parquet
```

Tên này rõ hơn và đúng đồ án hơn.

---

# 16. Output Stage 1/2 nên chuẩn hóa

## Stage 1 output

```text
data/processed/ohlcv.parquet
data/processed/data_summary.json
```

`ohlcv.parquet` thực chất là OHLCV+, nhưng tên vẫn chấp nhận được.

## Stage 2 output

```text
data/processed/features.parquet
data/processed/features.feature_list.json
data/processed/labels.parquet
data/modeling/ml_dataset.parquet
data/modeling/label_distribution.json
data/modeling/feature_list.json
```

Nếu muốn gọn hơn:

```text
data/modeling/
  ml_dataset.parquet
  dataset_summary.json
  feature_list.json
```

Tôi thích phương án gọn hơn.

---

# 17. Chỉnh `constants.py`

Hiện `CORE_STATIC_FEATURES` đang 32 feature. ([GitHub][3])

Tôi khuyên đổi thành:

```python
CORE_STATIC_FEATURES: tuple[str, ...] = (
    # Momentum
    "return_1h",
    "return_4h",
    "rsi_14",
    "macd_hist_atr",

    # Trend
    "ema34_vs_ema89",
    "close_vs_ema_34",
    "adx_14",

    # Volatility
    "atr_pct_close",
    "atr_percentile",
    "high_low_range_20",

    # Position
    "price_position_20",
    "close_vs_vwap_atr",

    # Time/session
    "day_of_week",
    "sess_london",
    "sess_ny_am",

    # Tick-derived
    "spread_pct_close",
    "tick_count_zscore_20",
    "volume_imbalance",
)
```

Và bỏ `vwap` raw khỏi feature list.

Nên giữ `vwap` trong intermediate dataframe để tính `close_vs_vwap_atr`, nhưng không đưa `vwap` vào model.

---

# 18. Thứ tự ưu tiên sửa

## P0 — nên làm ngay

| Việc                                                                                | Lý do                        |
| ----------------------------------------------------------------------------------- | ---------------------------- |
| Thêm `bid_volume`, `ask_volume`, `volume_imbalance`, `spread_pct_close` vào Stage 1 | Tận dụng raw data            |
| Thay raw `vwap` bằng `close_vs_vwap_atr`                                            | Tránh feature non-stationary |
| Giảm `CORE_STATIC_FEATURES` từ 32 xuống 18                                          | Đỡ rối, dễ bảo vệ            |
| Bỏ Stochastic/Williams/lag features khỏi main set                                   | Giảm trùng lặp oscillator    |
| Đổi `build_features()` thành orchestration ngắn                                     | Tuân thủ rule code           |

## P1 — nên làm tiếp

| Việc                                                | Lý do                       |
| --------------------------------------------------- | --------------------------- |
| Tách `feature_blocks.py`                            | Giảm function dài           |
| Giảm comment section trong `build_features.py`      | Function name tự giải thích |
| Đổi Stage 2 wording thành `ML Dataset Construction` | Đúng thesis hơn             |
| Bỏ sample weight nếu Stage 3 không cần              | Giảm nhánh học thuật phụ    |

## P2 — nếu còn thời gian

| Việc                                      | Lý do                   |
| ----------------------------------------- | ----------------------- |
| Thêm ablation: 32 features vs 18 features | Có bằng chứng giản lược |
| Test 2-class Short/Long                   | Giảm nhiễu Hold         |
| Thêm report “feature set summary”         | Dễ bảo vệ               |

---

# 19. Chốt kiến trúc cuối

```text
Stage 1 — Market Data Preparation
raw tick parquet
→ validate quotes
→ microprice
→ OHLCV+ aggregation
→ date filter
→ data summary
→ ohlcv.parquet

Stage 2 — ML Dataset Construction
ohlcv.parquet
→ minimal causal features
→ triple-barrier labels
→ final ml_dataset.parquet
→ feature_list.json
→ label_distribution.json
```

Feature set chính:

```text
18 causal features
= price return + trend + volatility + position + time + tick-derived microstructure
```

Đây là hướng cân bằng nhất: **ít code hơn, feature dễ giải thích hơn, tận dụng đúng raw bid/ask data hơn, và đúng trọng tâm đồ án ML hơn.**

[1]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-small/src/thesis/pipeline.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-small/src/thesis/data/prepare_dataset.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-small/src/thesis/shared/constants.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-small/src/thesis/dataset/build_features.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/ultimateBroK/thesis-hybrid/hybrid-small/src/thesis/dataset/build_labels.py "raw.githubusercontent.com"
