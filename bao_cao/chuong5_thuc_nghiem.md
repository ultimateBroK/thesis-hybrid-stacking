# Chương 5. Thực nghiệm và đánh giá

## 5.1. Thiết lập thực nghiệm (Experimental Setup)

### 5.1.1. Môi trường thực nghiệm

| Thành phần | Chi tiết |
|---|---|
| Ngôn ngữ | Python 3.13 |
| Quản lý môi trường | Pixi (conda-based) |
| Xử lý dữ liệu | Polars (DataFrame), NumPy (numeric) |
| Deep Learning | PyTorch |
| Gradient Boosting | LightGBM |
| PCA | scikit-learn |
| Backtest | backtesting.py |
| Linting/Format | Ruff |
| Testing | pytest, coverage ≥ 60% |
| Random seed | 2024 (cố định cho khả năng tái lập) |

### 5.1.2. Dữ liệu thực nghiệm

- **Cặp tiền:** XAU/USD
- **Khung thời gian:** 1 giờ (1H)
- **Khoảng dữ liệu:** 2018-01-01 đến 2026-04-30
- **Tổng thanh giá:** ~72,000 (sau khi loại warm-up rows)

### 5.1.3. Phân chia dữ liệu

Pipeline sử dụng walk-forward sliding window (không phải fixed split):

| Tham số | Giá trị | Mô tả |
|---|---|---|
| Train window | 17,520 bars (~2 năm) | Số lượng thanh giá huấn luyện |
| Test window | 4,380 bars (~6 tháng) | Số lượng thanh giá kiểm tra |
| Step | 4,380 bars | Test windows không chồng chéo |
| Purge | 48 bars (2× horizon) | Khoảng loại bỏ cuối train |
| Embargo | 50 bars (> sequence_length) | Khoảng đệm đầu test |

Với dữ liệu ~72,000 thanh giá, pipeline tạo ra nhiều cửa sổ walk-forward, mỗi cửa sổ là một thực nghiệm độc lập.

### 5.1.4. So sánh mô hình

Các mô hình được tổ chức thành **4 nhóm** so sánh chính, nhằm tách biệt đóng góp của từng thành phần kiến trúc:

| Nhóm | Mô hình | Mục đích |
|---|---|---|
| **Nhóm 1: Naive Direction** | Naive Direction (persistence) + Always Long/Short/Hold, Majority Class, Random | Baseline tối thiểu (không học) |
| **Nhóm 2: LightGBM Static** | LightGBM chỉ dùng 22 đặc trưng tĩnh | Trần hiệu suất đặc trưng bảng |
| **Nhóm 3: GRU-only** | LightGBM chỉ dùng GRU hidden states | Trần hiệu suất đặc trưng tuần tự |
| **Nhóm 4: Hybrid GRU+LightGBM** | GRU PCA 16 + 22 đặc trưng tĩnh → LightGBM (mô hình đề xuất) | Hệ thống đầy đủ |

Chi tiết các baseline bổ sung trong Nhóm 1:

1. **Naive Direction** — dự đoán lặp lại hướng của thanh giá trước (persistence)
2. **Always Long** — luôn dự đoán Long (+1)
3. **Always Short** — luôn dự đoán Short (−1)
4. **Always Hold** — luôn dự đoán Hold (0)
5. **Majority Class** — luôn dự đoán lớp chiếm đa số trong tập huấn luyện
6. **Random** — dự đoán ngẫu nhiên từ {−1, 0, +1} với seed cố định

Tất cả mô hình chạy trên cùng các cửa sổ walk-forward, đảm bảo so sánh công bằng. Bộ baseline đa dạng này giúp đặt ngưỡng tham chiếu (floor/ceiling) cho hiệu suất ở nhiều chiều: Majority Class phản ánh bias dữ liệu, Always Long/Short kiểm tra thiên vị hướng, Naive Direction kiểm tra tính persistence của chuỗi.

## 5.2. Classification Metrics

### 5.2.1. Accuracy

$$\text{Accuracy} = \frac{\text{số dự đoán đúng}}{\text{tổng số mẫu}}$$

Accuracy đơn giản nhưng có thể gây hiểu lầm khi phân phối lớp mất cân bằng (ví dụ: Hold chiếm 60% → luôn dự đoán Hold cho accuracy 60%).

### 5.2.2. Macro F1-Score

$$F1_{\text{macro}} = \frac{1}{K} \sum_{c=1}^{K} F1_c$$

$$F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

Macro F1 tính F1 cho mỗi lớp rồi lấy trung bình — mỗi lớp có trọng số bằng nhau bất kể kích thước. Đây là metric chính cho đánh giá mô hình vì nó không bị ảnh hưởng bởi mất cân bằng lớp [1].

### 5.2.3. Directional Accuracy

$$\text{DirAcc} = \frac{\text{số mẫu dự đoán đúng hướng (Long/Short)}}{\text{tổng số mẫu có hướng (không phải Hold)}}$$

Directional accuracy chỉ xét các mẫu có hướng (+1 hoặc −1), bỏ qua Hold. Metric này quan trọng nhất trong bối cảnh giao dịch vì phản ánh khả năng mô hình nhận biết đúng hướng thị trường.

### 5.2.4. Per-class Precision, Recall, F1

Báo cáo chi tiết cho từng lớp:

| Metric | Ý nghĩa |
|---|---|
| Precision(Long) | Trong số dự đoán Long, bao nhiêu % thực sự tăng? |
| Recall(Long) | Trong số thực tế tăng, bao nhiêu % được phát hiện? |
| Precision(Short) | Trong số dự đoán Short, bao nhiêu % thực sự giảm? |
| Recall(Short) | Trong số thực tế giảm, bao nhiêu % được phát hiện? |

## 5.3. Regression Auxiliary Metrics

### 5.3.1. Mục đích

Dù mô hình chính là classification, regression auxiliary metrics cung cấp thêm thông tin về chất lượng dự báo:

- **Mean Absolute Error (MAE):** Sai số tuyệt đối trung bình của dự đoán regression.
- **Directional R²:** R² tính chỉ trên các mẫu có hướng (Long/Short), cho thấy mô hình giải thích được bao nhiêu variance của giá thay đổi.

### 5.3.2. Hạn chế

Regression metrics chỉ là phụ trợ (auxiliary) vì:
- Mục tiêu chính là phân loại đúng hướng, không phải dự báo giá tuyệt đối.
- Regression targets có thể có outliers lớn (flash crash, spike) làm sai lệch metric.

## 5.4. So sánh mô hình (Model Comparison)

### 5.4.1. OOF (Out-of-Fold) Aggregate

OOF predictions từ tất cả cửa sổ walk-forward được gom lại thành một chuỗi liên tục. Các metric aggregate bao gồm:

| Metric | Hybrid | Naive Dir. | Always Long | Always Short | Always Hold | Majority | Random |
|---|---|---|---|---|---|---|---|
| Accuracy | — | — | — | — | — | — | — |
| Macro F1 | — | — | — | — | — | — | — |
| Dir. Accuracy | — | — | — | — | — | — | — |

*(Kết quả cụ thể sẽ được điền sau khi chạy `pixi run workflow`)*

### 5.4.2. T-test thống kê

Để xác định sự khác biệt giữa các mô hình có ý nghĩa thống kê, sử dụng paired t-test trên per-window accuracies:

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

trong đó $\bar{d}$ là sự khác biệt trung bình, $s_d$ là độ lệch chuẩn, và $n$ là số cửa sổ. p-value < 0.05 cho thấy sự khác biệt có ý nghĩa thống kê [2].

## 5.5. Per-Window Stability

### 5.5.1. Phân tích ổn định

Mô hình tốt không chỉ có hiệu suất trung bình cao mà còn ổn định qua các cửa sổ thời gian. Phân tích per-window bao gồm:

- **Box plot accuracy per window:** Phân phối accuracy qua các cửa sổ.
- **Std accuracy:** Độ lệch chuẩn accuracy — thấp hơn = ổn định hơn.
- **Worst window:** Cửa sổ có hiệu suất thấp nhất — thường tương ứng với regime change đột ngột.
- **Best window:** Cửa sổ có hiệu suất cao nhất.

### 5.5.2. Phân tích theo chế độ thị trường

Mỗi cửa sổ test rơi vào một chế độ thị trường khác nhau:

- **Trending up:** ADX > 25, positive EMA slope → mô hình thường hoạt động tốt.
- **Trending down:** ADX > 25, negative EMA slope → mô hình thường hoạt động tốt.
- **Ranging:** ADX < 20 → mô hình thường gặp khó khăn, Hold nhiều.
- **Volatile:** ATR cao → barrier rộng, nhãn Hold nhiều, accuracy có thể thấp.

Phân tích theo regime giúp hiểu **khi nào mô hình hoạt động tốt/kém** và nguyên nhân.

## 5.6. Phân tích lỗi / Confusion Matrix

### 5.6.1. Confusion matrix aggregate

Confusion matrix tổng hợp trên toàn bộ OOF predictions:

```
              Predicted
              Short  Hold  Long
Actual Short  [ TN   FP    FP ]
       Hold   [ FN   TN    FP ]
       Long   [ FN   FN    TP ]
```

Các lỗi quan trọng cần phân tích:

- **False Long (thực tế Short, dự đoán Long):** Nguy hiểm nhất trong giao dịch — vào lệnh sai hướng.
- **False Short (thực tế Long, dự đoán Short):** Tương tự, sai hướng.
- **Hold confusion:** Dự đoán Hold khi thực tế có hướng → cơ hội bị bỏ lỡ (missed opportunity), không gây lỗ.

### 5.6.2. Phân tích lỗi theo thời gian

Xác định các giai đoạn mà mô hình dự đoán sai nhiều nhất:

- **Period 1:** Mô hình dự đoán tốt (high accuracy)
- **Period 2:** Mô hình dự đoán kém (low accuracy)

Nguyên nhân có thể:

- Regime change không được cửa sổ huấn luyện bao phủ.
- Dữ liệu kiểm tra có phân phối khác đáng kể so với huấn luyện.
- Sự kiện bất thường (flash crash, FOMC surprise) không thể dự đoán.

## 5.7. Feature Importance

### 5.7.1. Phương pháp

LightGBM cung cấp feature importance dựa trên split gain — tổng giảm impurity mà mỗi feature mang lại qua tất cả các cây [3]. Đồ án báo cáo:

- **Top-20 features quan trọng nhất** cho mỗi cửa sổ walk-forward.
- **Stability of importance:** Feature nào quan trọng ổn định qua các cửa sổ vs. chỉ quan trọng trong một số cửa sổ.

### 5.7.2. Phân loại feature importance

Feature importance được phân tích theo nhóm:

| Nhóm | Features ví dụ | Câu hỏi |
|---|---|---|
| GRU hidden states | gru_pc_0, gru_pc_1, ... | GRU có đóng góp thông tin tuần tự không? |
| Trend indicators | ema_slope, adx, ema_cross | Xu hướng có quan trọng không? |
| Volatility | atr, hl_range | Biến động có giúp phân loại không? |
| Oscillators | rsi, macd_hist | Dao động có hữu ích không? |
| Regime | regime | Chế độ thị trường có giúp không? |

Nếu GRU principal components xuất hiện trong top features → kiến trúc hybrid đang đóng góp giá trị. Nếu chỉ technical indicators → GRU có thể không cần thiết.

### 5.7.3. SHAP values (tùy chọn)

Để phân tích sâu hơn, SHAP (SHapley Additive exPlanations) [4] có thể được sử dụng để hiểu contribution của từng feature cho từng dự đoán cụ thể. Tuy nhiên, SHAP computation tốn kém nên chỉ áp dụng cho mẫu nhỏ.

## 5.8. Calibration

### 5.8.1. Mục đích

Calibration đo lường xem xác suất dự đoán có phản ánh đúng xác suất thực tế không [5]. Ví dụ: nếu mô hình dự đoán "70% xác suất Long", thì trong số tất cả các dự đoán 70% này, thực sự ~70% nên là Long.

### 5.8.2. Reliability diagram

Biểu đồ reliability diagram vẽ predicted probability (trục x) vs. observed frequency (trục y). Đường chéo 45° = perfectly calibrated.

- **Over-confident:** Đường nằm dưới chéo → mô hình quá tự tin.
- **Under-confident:** Đường nằm trên chéo → mô hình thiếu tự tin.

### 5.8.3. Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |p_b - \text{acc}_b|$$

trong đó $B$ là số bins, $n_b$ là số mẫu trong bin $b$, $p_b$ là xác suất trung bình dự đoán, và $\text{acc}_b$ là accuracy thực tế trong bin.

## 5.9. Tổng kết chương

Chương này đã trình bày thiết lập thực nghiệm, các metric đánh giá (classification, regression auxiliary), phương pháp so sánh mô hình, phân tích ổn định per-window, phân tích lỗi qua confusion matrix, feature importance, và calibration. Kết quả cụ thể sẽ được điền sau khi chạy pipeline hoàn chỉnh.

## Tài liệu tham khảo chương này

[1] Krauss, C., et al. (2017). "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689–702.

[2] Arnott, R., Harvey, C.R., & Markowitz, H.M. (2019). "A Backtesting Protocol in the Era of Machine Learning." *Journal of Financial Data Science*, 1(1), 64–74.

[3] Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NIPS 30*, pp. 3149–3157.

[4] Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems 30*.

[5] Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML 2017*.
