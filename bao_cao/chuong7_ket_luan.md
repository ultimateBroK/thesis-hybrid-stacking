# Chương 7. Kết luận

## 7.1. Tổng kết kết quả đạt được

Đồ án này đã xây dựng và đánh giá hệ thống dự báo xu hướng ngắn hạn XAU/USD sử dụng kiến trúc hybrid GRU + LightGBM. Các kết quả chính bao gồm:

### 7.1.1. Về mặt hệ thống

1. **Pipeline hoàn chỉnh:** Xây dựng pipeline end-to-end từ thu thập dữ liệu đến báo cáo kết quả, bao gồm 6 giai đoạn: data processing → feature engineering → labeling → model training → backtest → reporting.

2. **Kiến trúc hybrid:** Triển khai thành công kiến trúc stacking GRU + LightGBM, trong đó GRU đóng vai trò trích xuất đặc trưng tuần tự (với attention pooling và variational dropout), PCA giảm chiều, và LightGBM phân loại trên ma trận đặc trưng lai.

3. **Triple Barrier labeling:** Áp dụng phương pháp gán nhãn Triple Barrier với ATR-based dynamic barriers, average-uniqueness sample weighting, và xử lý nhãn censored/ambiguous — phù hợp với đặc thù bài toán giao dịch.

4. **Walk-forward validation:** Triển khai validation trượt với event-time purge và embargo, đảm bảo không rò rỉ thông tin giữa tập huấn luyện và kiểm tra — kể cả cho chuỗi GRU (sequence overlap).

### 7.1.2. Về mặt kỹ thuật

5. **Đa lớp chống leakage:** 8 biện pháp chống rò rỉ thông tin ở các cấp: label, feature, normalization, dimensionality, model, và sample weighting.

6. **Feature engineering có chủ ý:** 5 nhóm đặc trưng (price/volatility, trend, oscillators, regime, normalized OHLCV) với lọc tương quan và loại warm-up rows.

7. **Đánh giá toàn diện:** Classification metrics (accuracy, macro F1, directional accuracy), regression auxiliary metrics, per-window stability analysis, confusion matrix, feature importance, và calibration.

### 7.1.3. Về mặt học thuật

8. **So sánh với baseline:** Mô hình hybrid được so sánh với naive persistence, majority class, và random baseline trên cùng các cửa sổ walk-forward.

9. **Code chất lượng cao:** Toàn bộ source code được lint (Ruff), test (pytest, coverage ≥ 60%), và tổ chức module hóa theo stage pattern.

## 7.2. Hạn chế

### 7.2.1. Hạn chế của dữ liệu

1. **Độ phủ thị trường:** Dữ liệu chỉ bao gồm XAU/USD trên một nguồn duy nhất. Các thị trường khác (equity, crypto, commodities) có thể có đặc điểm khác.

2. **Volume data:** Khối lượng giao dịch trong forex không centralized — volume data có thể không phản ánh chính xác tổng khối lượng thị trường.

3. **Chỉ OHLCV:** Không sử dụng dữ liệu bổ sung như order book depth, sentiment, macro economic indicators, hoặc inter-market correlations.

### 7.2.2. Hạn chế của mô hình

4. **Chỉ GRU:** Chưa thử nghiệm các kiến trúc khác như LSTM, Transformer, hoặc Temporal Convolutional Network (TCN) làm feature extractor.

5. **PCA tuyến tính:** PCA giả định tuyến tính trong việc giảm chiều hidden states — có thể mất thông tin phi tuyến. Tùy chọn: autoencoder, t-SNE, UMAP.

6. **Không tuning hyperparameter:** Các hyperparameter (learning rate, num_leaves, hidden_size, v.v.) được chọn dựa trên kinh nghiệm, không qua systematic optimization (Optuna, Grid Search). Điều này có nghĩa hiệu suất có thể chưa tối ưu.

7. **Không có meta-labeling:** Chưa áp dụng meta-labeling [1] (mô hình thứ hai quyết định có theo tín hiệu của mô hình chính hay không) — có thể cải thiện precision.

### 7.2.3. Hạn chế của phương pháp luận

8. **Horizon cố định:** Triple Barrier sử dụng horizon cố định 24 giờ. Trong thực tế, thời gian giữ vị thế tối ưu thay đổi theo chế độ thị trường.

9. **Không đánh giá giao dịch thực:** Backtest chỉ là minh họa, không có forward test trên dữ liệu real-time hoặc paper trading.

10. **Không so sánh với mô hình SOTA:** Chưa so sánh với các mô hình state-of-the-art như Temporal Fusion Transformer (TFT), PatchTST, hoặc iTransformer.

## 7.3. Hướng phát triển

### 7.3.1. Ngắn hạn

1. **Hyperparameter optimization:** Sử dụng Optuna [2] để tối ưu hóa hyperparameter cho cả GRU và LightGBM trong walk-forward framework.

2. **Thêm kiến trúc GRU:** Thử nghiệm bidirectional GRU, multi-head attention, và stacked GRU với skip connections.

3. **Meta-labeling:** Thêm mô hình meta-labeling (level 2) quyết định có act trên tín hiệu của mô hình hybrid hay không.

4. **Nhiều chỉ báo kỹ thuật hơn:** Thêm các chỉ báo như Ichimoku, Bollinger Band width, VWAP, và các đặc trưng thống kê bậc cao (skewness, kurtosis rolling).

### 7.3.2. Trung hạn

5. **Multi-asset:** Mở rộng sang các cặp tiền khác (EUR/USD, GBP/JPY) hoặc tài sản khác (S&P 500, Bitcoin) để đánh giá khả năng tổng quát hóa.

6. **Multi-timeframe:** Sử dụng đồng thời dữ liệu từ nhiều khung thời gian (M15, 1H, 4H, 1D) — cho phép mô hình nắm bắt cả cấu trúc vi mô và vĩ mô.

7. **Transformer-based feature extractor:** Thay thế GRU bằng Temporal Fusion Transformer (TFT) [3] hoặc PatchTST [4] — các kiến trúc SOTA cho time-series forecasting.

8. **Dynamic horizon:** Thay vì horizon cố định 24h, sử dụng dynamic horizon dựa trên ATR regime — barrier rộng hơn trong thị trường biến động, hẹp hơn trong thị trường bình ổn.

### 7.3.3. Dài hạn

9. **Online learning:** Triển khai mô hình học liên tục (online/incremental learning) thay vì retrain theo batch — cho phép thích ứng nhanh hơn với regime change.

10. **Explainability:** Tích hợp SHAP values và attention visualization để cung cấp giải thích trực quan cho từng dự đoán.

11. **Paper trading / forward test:** Triển khai trên tài khoản demo real-time để đánh giá hiệu suất trong điều kiện thị trường thực (latency, slippage thực, tâm lý).

12. **Risk management nâng cao:** Tích hợp Kelly criterion, CVaR optimization, hoặc reinforcement learning cho position sizing thay vì fixed lot size.

## 7.4. Kết luận

Đồ án này đã trình bày một phương pháp dự báo xu hướng ngắn hạn XAU/USD dựa trên kiến trúc hybrid GRU + LightGBM với Triple Barrier labeling và walk-forward validation. Kiến trúc stacking kết hợp thế mạnh nắm bắt mẫu hình tuần tự của GRU với khả năng xử lý đặc trưng bảng phi tuyến của LightGBM, trong khi các biện pháp chống leakage đa lớp đảm bảo tính hợp lệ của kết quả đánh giá.

Kết quả thực nghiệm cho thấy mô hình hybrid có tiềm năng trong bài toán phân loại xu hướng, nhưng cũng bộc lộ những hạn chế nhất định — đặc biệt trong các giai đoạn regime change đột ngột và thị trường dao động ngang. Điều này phù hợp với kỳ vọng học thuật: dự báo tài chính là bài toán intrinsically khó, và không có mô hình đơn lẻ nào có thể giải quyết hoàn toàn [1, 5].

Đóng góp chính của đồ án không phải là một chiến lược giao dịch sinh lời, mà là một framework đánh giá nghiêm ngặt cho mô hình hybrid trong bối cảnh tài chính — với emphasis trên phương pháp luận đúng (proper validation, anti-leakage, honest evaluation) hơn là kết quả backtest lạc quan.

## Tài liệu tham khảo chương này

[1] López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

[2] Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD 2019*.

[3] Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." *International Journal of Forecasting*, 37(4), 1748–1764.

[4] Nie, Y., et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR 2023*.

[5] López de Prado, M. (2018). "The 10 Reasons Most Machine Learning Funds Fail." *Journal of Portfolio Management*, 44(6), 120–133.
