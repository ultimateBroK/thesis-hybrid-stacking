# Chương 4. Phương pháp đề xuất

## 4.1. Tổng quan kiến trúc

Hệ thống được thiết kế theo dạng pipeline tuần tự gồm 6 giai đoạn (stage):

```
Stage 1: Thu thập & tổng hợp dữ liệu (Data Processing)
    ↓
Stage 2: Tạo đặc trưng (Feature Engineering)
    ↓
Stage 3: Gán nhãn (Labeling — Triple Barrier)
    ↓
Stage 4: Huấn luyện mô hình (Training — Walk-Forward Hybrid)
    ↓
Stage 5: Backtest minh họa (Backtest Demo)
    ↓
Stage 6: Báo cáo kết quả (Reporting)
```

Chương này tập trung vào các quyết định thiết kế ở Stage 3 và Stage 4 — phần cốt lõi của phương pháp đề xuất.

## 4.2. Thiết kế nhãn (Label Design)

### 4.2.1. Thông số Triple Barrier

Nhãn được tạo bằng phương pháp Triple Barrier (Chương 2, mục 2.6) với các thông số:

| Thông số | Giá trị | Giải thích |
|---|---|---|
| `atr_tp_multiplier` | 2.0 | Take-profit = 2 × ATR |
| `atr_sl_multiplier` | 2.0 | Stop-loss = 2 × ATR |
| `horizon_bars` | 24 | Vertical barrier = 24 giờ |
| `min_atr` | (tự động) | Floor ATR tối thiểu |

**Lý do chọn thông số:**

- **TP/SL bằng nhau (2.0 × ATR):** Nhắm đến risk-reward ratio 1:1, đồng thời đảm bảo phân phối nhãn cân bằng hơn giữa Long và Short. Nếu TP/SL không đối xứng, barrier dễ chạm hơn (nhỏ hơn) sẽ tạo ra nhiều nhãn về phía đó, khiến mô hình bị thiên vị hướng dự đoán [1].
- **Horizon 24 giờ:** Tương ứng với khoảng thời gian giữ vị thế 1 ngày giao dịch — phù hợp với mục tiêu dự báo ngắn hạn.
- **ATR-based barriers:** Thích ứng tự động với biến động thị trường, đảm bảo barrier có ý nghĩa kinh tế tương đương giữa các chế độ thị trường [2].

### 4.2.2. Xử lý nhãn ambiguous

Khi cả upper barrier và lower barrier đều bị chạm trong cùng một thanh giá (same-bar both-hit), OHLCV không cho biết đường giá đi theo thứ tự nào. Các mẫu này được gán nhãn Hold (0) — quyết định bảo thủ, tránh tạo nhãn sai [2].

### 4.2.3. Xử lý nhãn censored

Các mẫu trong khoảng `horizon_bars` (24 giờ) cuối chuỗi dữ liệu không có đủ dữ liệu tương lai để đánh giá barrier. Nếu giữ lại và gán nhãn Hold, sẽ tạo nhiễu nhãn (label noise). Thay vào đó, chúng được đánh dấu censored (−2) và **loại bỏ hoàn toàn** khỏi tập huấn luyện.

### 4.2.4. Sample weighting

Trọng số mẫu (sample weight) được tính bằng average uniqueness theo López de Prado (mục 2.6.3). Trọng số được chuẩn hóa để giá trị trung bình = 1, giữ nguyên scale của hàm mất mát:

```python
# Normalized to mean 1
weights /= weights.mean()
```

Trọng số này được sử dụng cả cho GRU (trong training loss) và LightGBM (qua `sample_weight` parameter).

### 4.2.5. Kiểm tra tính khả thi kinh tế

Sau khi tạo nhãn, pipeline kiểm tra tỷ lệ nhãn Long/Short thực sự có lãi sau chi phí giao dịch. Nếu cả hai lớp đều dưới 60%, cảnh báo được phát ra — nhãn có thể không hữu ích cho giao dịch thực tế.

## 4.3. Thiết kế Validation

### 4.3.1. Walk-forward sliding window

Thiết kế validation sử dụng sliding walk-forward với các thông số:

| Thông số | Giá trị | Tương đương |
|---|---|---|
| `train_window_bars` | 17,520 | ~2 năm trên khung 1H |
| `test_window_bars` | 4,380 | ~6 tháng trên khung 1H |
| `step_bars` | 4,380 | Không chồng chéo |
| `purge_bars` | 48 | 2× horizon (24h) |
| `embargo_bars` | 50 | Khoảng đệm bổ sung |
| `min_train_bars` | 10,000 | Tối thiểu ~14 tháng |

**Ví dụ cửa sổ:**

```
Window 1: Train [2018-01 → 2020-06] → Purge → Embargo → Test [2020-07 → 2020-12]
Window 2: Train [2020-01 → 2022-06] → Purge → Embargo → Test [2022-07 → 2022-12]
Window 3: Train [2022-01 → 2024-06] → Purge → Embargo → Test [2024-07 → 2024-12]
...
```

Mỗi cửa sổ kiểm tra (test window) không chồng chéo với cửa sổ trước, tạo ra các đánh giá out-of-sample độc lập.

### 4.3.2. Event-time purge

Thay vì purge cố định 48 thanh giá, đồ án sử dụng **event-time purge**: chỉ giữ lại các mẫu huấn luyện có `event_end < test_start`. Phương pháp này chính xác hơn vì:

- Nhãn Triple Barrier có thời gian sự kiện biến đổi (touched_bar từ 1 đến 24 giờ).
- Purge cố định có thể quá khắc nghiệt (loại bỏ mẫu đã kết thúc sự kiện) hoặc quá lỏng (giữ mẫu chưa kết thúc sự kiện).
- Event-time purge loại bỏ chính xác các mẫu có rủi ro label lookahead [2].

### 4.3.3. Embargo cho chống sequence leakage

Khoảng embargo (50 thanh giá ≈ 2 ngày) đặc biệt quan trọng cho GRU vì:

- GRU sử dụng chuỗi đầu vào dài 48 thanh giá (sequence_length = 48).
- Nếu test sample tại vị trí $t$ sử dụng chuỗi $[t-47, ..., t]$, và mẫu train tại $t-48$ được giữ lại → chuỗi test có chứa dữ liệu từ tập train.
- Embargo 50 thanh giá > sequence_length 48 → đảm bảo không có sequence overlap [2, 3].

Pipeline kiểm tra điều kiện này và raise error nếu `embargo_bars < sequence_length`:

```python
if gap_bars < seq_len:
    raise ValueError("Leakage risk: gap < sequence_length")
```

### 4.3.4. Chia validation nội bộ (Internal validation split)

Trong mỗi cửa sổ huấn luyện, 20% cuối được dành làm internal validation cho:

- **GRU early stopping:** Dừng huấn luyện nếu validation loss không cải thiện sau 20 epoch.
- **LightGBM early stopping:** Dừng boosting nếu validation loss không cải thiện sau 40 round.
- **GRU hidden state PCA:** Fit PCA chỉ trên phần train (80% đầu), transform cả train và internal val.

## 4.4. Mô hình Baseline

### 4.4.1. Khung so sánh 4 nhóm mô hình

Để đánh giá tương đối hiệu suất mô hình hybrid, đồ án sử dụng khung so sánh **4 nhóm** nhằm tách biệt đóng góp của từng thành phần:

| Nhóm | Kiến trúc | Mục đích |
|---|---|---|
| **Nhóm 1: Naive Direction** | Dự đoán lặp lại hướng thanh giá trước (persistence) | Baseline tối thiểu (không học) |
| **Nhóm 2: LightGBM Static** | 22 đặc trưng tĩnh chỉ LightGBM | Trần hiệu suất đặc trưng bảng |
| **Nhóm 3: GRU-only** | Chỉ hidden states GRU | Trần hiệu suất đặc trưng tuần tự |
| **Nhóm 4: Hybrid GRU+LightGBM** | GRU PCA 16 + 22 đặc trưng tĩnh → LightGBM | Hệ thống đầy đủ (đóng góp luận văn) |

Ngoài 4 nhóm chính, các chiến lược baseline bổ sung được sử dụng làm tham chiếu phụ trong Nhóm 1:

**Always Long / Always Short / Always Hold:** Luôn dự đoán một hướng cố định. Kiểm tra thiên vị hướng trong dữ liệu.

**Majority Class:** Luôn dự đoán lớp phổ biến nhất trong tập huấn luyện. Nếu Hold chiếm đa số → mô hình có accuracy cao nhưng không có giá trị giao dịch. Phản ánh bias dữ liệu.

**Random Baseline:** Dự đoán ngẫu nhiên từ {−1, 0, 1} với seed cố định. Thiết lập floor cho hiệu suất.

**Naive Direction (Persistence):** Dự đoán hướng của thanh giá trước đó:

$$\hat{y}_t = \text{sign}(r_{t-1})$$

Giả định rằng xu hướng gần nhất sẽ tiếp tục — baseline tối thiểu cho bài toán prediction.

### 4.4.2. Đánh giá baseline

Mỗi baseline được chạy trên cùng các cửa sổ walk-forward với mô hình hybrid, đảm bảo so sánh công bằng. Các metric: accuracy, macro F1, directional accuracy.

## 4.5. Kiến trúc Hybrid GRU + LightGBM

### 4.5.1. Giai đoạn 1: GRU Feature Extraction

**Chuẩn bị chuỗi đầu vào:**

Từ feature matrix, GRU sử dụng chuỗi 48 thanh giá liên tiếp của các đặc trưng OHLCV chuẩn hóa (normalized OHLCV). Chuỗi được tạo bằng sliding window:

$$X_i^{\text{GRU}} = [x_{i-47}, x_{i-46}, ..., x_i] \in \mathbb{R}^{48 \times d_{\text{input}}}$$

trong đó $d_{\text{input}}$ là số đặc trưng GRU (normalized OHLCV columns). Mỗi đặc trưng được chuẩn hóa bằng mean và std tính trên tập huấn luyện:

$$\hat{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

**Kiến trúc GRU:**

| Layer | Thông số |
|---|---|
| Input Normalization | LayerNorm |
| Variational Dropout | p = 0.1 |
| GRU | 2 layers, hidden_size = 64, dropout = 0.3 |
| Attention Pooling | Linear(64 → 1) + Softmax |
| Output | Context vector (64 chiều) |

Quy trình forward:

1. LayerNorm chuẩn hóa đầu vào.
2. Variational dropout áp dụng mask đồng nhất trên toàn bộ chuỗi.
3. GRU 2 lớp xử lý chuỗi, tạo hidden state cho mỗi bước thời gian.
4. Attention pooling tính trọng số cho mỗi bước và tổng hợp thành context vector duy nhất.
5. Context vector (64 chiều) là đầu ra của giai đoạn GRU.

**Huấn luyện GRU:**

- Optimizer: Adam, learning rate = 0.0005
- Batch size: 256
- Max epochs: 100, patience = 20 (early stopping)
- Warmup: 3 epoch (linear ramp-up learning rate)
- Gradient accumulation: 1 step

GRU được huấn luyện như một classifier độc lập (multiclass: −1, 0, 1) để đảm bảo hidden states mang thông tin phân loại hữu ích. Validation split nội bộ (20%) dùng cho early stopping.

### 4.5.2. Giảm chiều PCA

Hidden states GRU (64 chiều) thường chứa nhiều nhiễu do mạng nơ-ron có xu hướng sử dụng không gian chiều cao không hoàn toàn [4]. PCA giảm xuống 16 thành phần chính:

```python
PCA(n_components=16)
```

PCA được fit chỉ trên tập huấn luyện (80% đầu) để tránh rò rỉ thông tin từ validation. Explained variance được ghi nhận:

```
GRU hidden states: 64→16 PCs, explained variance=XX%
```

Nếu explained variance < 50%, cảnh báo được phát ra — không gian hidden state chủ yếu là nhiễu.

### 4.5.3. Giai đoạn 2: Xây dựng ma trận đặc trưng lai (Hybrid Feature Matrix)

Ma trận đặc trưng cho LightGBM được xây dựng bằng cách ghép (concatenate):

$$X^{\text{hybrid}} = [\text{GRU\_PCs} \| \text{Static\_Features}]$$

- **GRU_PCs:** 16 thành phần chính từ hidden states GRU.
- **Static_Features:** Các đặc trưng kỹ thuật đã lọc (RSI, ATR, ADX, MACD, EMA slope, returns, regime, v.v.).

Ma trận lai có kích thước $N \times (16 + d_{\text{static}})$.

Alignment quan trọng: GRU chuỗi đầu vào bắt đầu từ sample $i-47$, nên hidden state tại sample $i$ chỉ tồn tại từ sample 47 trở đi. Pipeline tự động align:

```python
train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
```

### 4.5.4. Giai đoạn 3: Phân loại LightGBM

**Thông số LightGBM:**

| Thông số | Giá trị | Giải thích |
|---|---|---|
| `num_leaves` | 31 | Số lá tối đa |
| `max_depth` | 6 | Độ sâu tối đa |
| `learning_rate` | 0.02 | Shrinkage |
| `n_estimators` | 500 | Số vòng boosting tối đa |
| `min_child_samples` | 50 | Regularization |
| `subsample` | 0.80 | Row subsampling |
| `feature_fraction` | 0.70 | Column subsampling |
| `reg_alpha` | 0.05 | L1 regularization |
| `reg_lambda` | 5.0 | L2 regularization |
| `early_stopping` | 40 rounds | Dừng sớm |

**Class weighting:** Trọng số lớp được tính tự động để bù đắp mất cân bằng:

$$w_c = \frac{N}{K \cdot n_c}$$

trong đó $N$ là tổng số mẫu, $K$ là số lớp, và $n_c$ là số mẫu trong lớp $c$.

**Distribution shift weighting:** Vì mỗi cửa sổ walk-forward có thể có phân phối lớp khác nhau, trọng số shift được tính:

$$w_{\text{shift}}(c) = \frac{p_{\text{train}}(c)}{p_{\text{val}}(c)}$$

Trọng số này được nhân với average-uniqueness sample weight:

$$w_{\text{combined}} = w_{\text{uniqueness}} \times w_{\text{shift}}$$

### 4.5.5. Dự đoán và thu thập OOF

Mỗi cửa sổ walk-forward tạo ra dự đoán trên tập kiểm tra (out-of-fold predictions). Các OOF predictions được gom lại thành một chuỗi dự đoán liên tục — cho phép đánh giá hiệu suất tổng thể trên toàn bộ dữ liệu out-of-sample.

## 4.6. Biện pháp chống rò rỉ thông tin (Anti-Leakage)

### 4.6.1. Tổng hợp các biện pháp

Đồ án triển khai nhiều lớp bảo vệ chống rò rỉ thông tin [2, 3, 5]:

| Lớp | Biện pháp | Vai trò |
|---|---|---|
| Label | Censored label removal | Loại bỏ mẫu thiếu forward data |
| Label | Event-time purge | Train labels kết thúc trước test boundary |
| Sequence | Embargo ≥ sequence_length | GRU chuỗi không overlap test |
| Feature | Train-only feature selection | Correlation filter chỉ fit trên train |
| Feature | Train-only normalization | Mean/std chỉ từ train set |
| Dimensionality | Train-only PCA | PCA fit chỉ trên train hidden states |
| Model | Train-only early stopping | Val split nội bộ từ train window |
| Sample | Average-uniqueness weighting | Giảm bias từ overlapping labels |

### 4.6.2. Tại sao cần nhiều lớp?

Rò rỉ thông tin trong tài chính thường tinh vi và khó phát hiện [5]. Một số dạng phổ biến:

- **Label lookahead:** Nhãn tại $t$ dùng data từ $t$ đến $t+h$; nếu $t+h$ nằm trong test set → leakage. Purge + embargo giải quyết.
- **Feature lookahead:** Feature selection trên toàn bộ data trước khi split → leakage. Train-only selection giải quyết.
- **Normalization leakage:** Tính mean/std trên cả train + test → leakage. Train-only normalization giải quyết.
- **Sequence overlap:** GRU input tại test sample $t$ chứa data từ train period. Embargo giải quyết.

## 4.7. Tổng kết chương

Chương này đã trình bày phương pháp đề xuất với các đóng góp chính: (1) thiết kế nhãn Triple Barrier với ATR-based barriers và sample weighting; (2) walk-forward validation với event-time purge và embargo chống sequence leakage; (3) kiến trúc hybrid GRU (feature extractor) + LightGBM (classifier) với PCA giảm chiều; và (4) đa lớp bảo vệ chống rò rỉ thông tin. Chương tiếp theo sẽ trình bày kết quả thực nghiệm.

## Tài liệu tham khảo chương này

[1] Kim, H., et al. (2025). "Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data." *arXiv:2504.02249*.

[2] López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

[3] Kapoor, S., et al. (2023). "On Leakage in Machine Learning Pipelines." *arXiv:2311.04179*.

[4] Gong, X., et al. (2025). "A Hybrid Data Mining Framework for Financial Time-Series Prediction." *Research Square preprint*.

[5] Bailey, D.H., et al. (2014). "Pseudo-Mathematics and Financial Charlatanism." *Notices of the AMS*, 61(5), 458–471.
