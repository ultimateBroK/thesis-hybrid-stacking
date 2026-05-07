# Chương 1. Giới thiệu

## 1.1. Đặt vấn đề

Thị trường tài chính toàn cầu hoạt động liên tục 24 giờ mỗi ngày với khối lượng giao dịch khổng lồ, tạo ra các chuỗi thời gian giá phức tạp và nhiều nhiễu. Trong bối cảnh đó, dự báo xu hướng giá ngắn hạn — đặc biệt là phân loại hướng di chuyển giá (tăng, giảm, hoặc đi ngang) — là một bài toán có ý nghĩa thực tiễn lớn đối với các nhà giao dịch và nhà quản lý rủi ro. Khác với dự báo giá tuyệt đối (regression), phân loại xu hướng (classification) cho phép các nhà đầu tư đưa ra quyết định giao dịch rõ ràng: mua (long), bán (short), hoặc giữ vị thế (hold) [1, 2].

Các phương pháp thống kê truyền thống như ARIMA hay GARCH đã được sử dụng rộng rãi trong nhiều thập kỷ để mô hình hóa chuỗi thời gian tài chính [3]. Tuy nhiên, các mô hình này thường giả định tuyến tính và phân phối chuẩn — hai đặc điểm hiếm khi đúng trong dữ liệu tài chính thực tế. Những năm gần đây, học sâu (deep learning) và học máy (machine learning) đã cho thấy hiệu quả vượt trội trong việc nắm bắt các mối quan hệ phi tuyến và mẫu hình phức tạp trong dữ liệu giá [4, 5].

Đặc biệt, các kiến trúc mạng hồi tiếp (recurrent neural networks) như LSTM (Long Short-Term Memory) và GRU (Gated Recurrent Unit) đã được chứng minh là có khả năng học các phụ thuộc thời gian dài hạn trong chuỗi thời gian tài chính [6, 7]. Song song đó, các mô hình ensemble dựa trên cây quyết định như LightGBM (Light Gradient Boosting Machine) [8] thể hiện sức mạnh đáng kể trong việc xử lý các đặc trưng dạng bảng (tabular features) với tốc độ huấn luyện nhanh và khả năng tổng hợp hóa tốt.

Một hướng tiếp cận đầy tiềm năng là kết hợp thế mạnh của cả hai họ mô hình này: sử dụng GRU để trích xuất đặc trưng tuần tự từ chuỗi giá, sau đó sử dụng các đặc trưng này làm đầu vào bổ sung cho LightGBM. Phương pháp hybrid stacking này đã cho thấy kết quả đáng khích lệ trong các nghiên cứu gần đây [9, 10].

## 1.2. Lý do chọn cặp tiền XAU/USD

XAU/USD (vàng so với đồng đô la Mỹ) là một trong những công cụ tài chính được giao dịch nhiều nhất trên thế giới, với khối lượng giao dịch hàng ngày vượt 183 tỷ USD vào năm 2024 [11]. Việc chọn XAU/USD làm đối tượng nghiên cứu dựa trên các lý do sau:

**Tính chất đặc thù của vàng:** Vàng vừa là tài sản trú ẩn an toàn (safe-haven asset), vừa là hàng hóa công nghiệp và phương tiện dự trữ giá trị. Giá vàng chịu ảnh hưởng bởi nhiều yếu tố vĩ mô bao gồm lãi suất của Fed (Cục Dự trữ Liên bang Mỹ), lạm phát, địa chính trị và sức mạnh đồng USD [12]. Đặc điểm này tạo ra các mẫu hình giá phức tạp, vừa có xu hướng (trending) vừa có tính dao động (volatile), phù hợp để đánh giá khả năng của mô hình hybrid.

**Tính thanh khoản và độ sẵn có dữ liệu:** XAU/USD được giao dịch gần như liên tục từ thứ Hai đến thứ Sáu trên nhiều sàn giao dịch toàn cầu. Điều này cung cấp nguồn dữ liệu OHLCV (Open-High-Low-Close-Volume) phong phú, ít khoảng trống (gap) và có độ tin cậy cao.

**Khoảng thời gian nghiên cứu:** Dữ liệu sử dụng trong đồ án trải dài từ tháng 1/2018 đến tháng 4/2026, bao gồm nhiều sự kiện thị trường quan trọng: đại dịch COVID-19 (2020), khủng hoảng lạm phát toàn cầu (2022–2023), và các chu kỳ lãi suất của Fed. Sự đa dạng về chế độ thị trường (market regime) giúp đánh giá tính ổn định của mô hình một cách toàn diện.

## 1.3. Mục tiêu nghiên cứu

Đồ án này tập trung vào **đánh giá hiệu suất dự báo xu hướng ngắn hạn** của mô hình hybrid GRU + LightGBM trên cặp XAU/USD với khung thời gian 1 giờ (1H). Các mục tiêu cụ thể bao gồm:

1. **Xây dựng pipeline dự báo xu hướng hoàn chỉnh:** Từ thu thập dữ liệu, tạo đặc trưng (feature engineering), gán nhãn theo phương pháp Triple Barrier [13], huấn luyện mô hình, đến đánh giá kết quả.

2. **So sánh hiệu suất mô hình hybrid với các nhóm baseline:** Khung so sánh 4 nhóm — Naive Direction (baseline tối thiểu), LightGBM Static (baseline đặc trưng bảng), GRU-only (baseline đặc trưng tuần tự), và Hybrid GRU+LightGBM (mô hình đề xuất). Các chiến lược phụ (Always Long/Short/Hold, Majority Class, Random) được sử dụng làm tham chiếu bổ sung.

3. **Đánh giá tính ổn định qua nhiều cửa sổ thời gian (walk-forward):** Sử dụng validation trượt với purge và embargo để đảm bảo không rò rỉ thông tin giữa tập huấn luyện và tập kiểm tra [13, 14].

4. **Phân tích chi tiết kết quả dự báo:** Bao gồm confusion matrix, feature importance, calibration, và phân tích lỗi theo từng cửa sổ thời gian.

**Lưu ý quan trọng:** Đồ án này **không** nhằm mục đích xây dựng hoặc tối ưu hóa một hệ thống giao dịch tự động (trading system). Phần minh họa backtest ở Chương 6 chỉ đóng vai trò tham khảo để hiểu cách mô hình có thể được ứng dụng, không phải bằng chứng chính cho hiệu quả kinh tế của phương pháp.

## 1.4. Đóng góp của đồ án

1. Đề xuất và triển khai kiến trúc hybrid stacking GRU + LightGBM, trong đó GRU đóng vai trò trích xuất đặc trưng tuần tự (sequence feature extractor) với attention pooling, và LightGBM đóng vai trò phân loại cuối cùng trên ma trận đặc trưng kết hợp (hidden states + technical indicators).

2. Áp dụng phương pháp gán nhãn Triple Barrier với ATR-based dynamic barriers và average-uniqueness sample weighting theo López de Prado [13], phù hợp với bài toán giao dịch có thời gian giữ vị thế biến đổi.

3. Triển khai walk-forward validation với event-time purge và embargo, ngăn chặn các dạng rò rỉ thông tin đặc thù của dữ liệu tài chính (label lookahead, sequence overlap).

4. Phân tích toàn diện kết quả dự báo trên nhiều cửa sổ out-of-sample, bao gồm per-window stability analysis và error analysis.

## 1.5. Bố cục đồ án

Đồ án được tổ chức thành 7 chương:

- **Chương 1** (chương này): Giới thiệu bài toán, mục tiêu và đóng góp.
- **Chương 2:** Trình bày cơ sở lý thuyết về time-series forecasting, mạng GRU, LightGBM, hybrid stacking, walk-forward validation và Triple Barrier labeling.
- **Chương 3:** Mô tả nguồn dữ liệu, quy trình tiền xử lý, feature engineering và kiểm tra chất lượng dữ liệu.
- **Chương 4:** Trình bày phương pháp đề xuất: thiết kế nhãn, thiết kế validation, mô hình baseline, kiến trúc hybrid và các biện pháp chống rò rỉ thông tin.
- **Chương 5:** Trình bày kết quả thực nghiệm, so sánh mô hình, phân tích ổn định và phân tích lỗi.
- **Chương 6:** Minh họa ứng dụng qua backtest demo và thảo luận hạn chế.
- **Chương 7:** Tổng kết, hạn chế và hướng phát triển.

## 1.6. Tài liệu tham khảo chương này

[1] Zhang, C., Sjarif, N.N.A., & Ibrahim, R. (2024). "Deep learning models for price forecasting of financial time series: A review of recent advancements: 2020–2022." *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 14(1), e1519.

[2] Lim, W.L. & Tsiakas, I. (2024). "Deep Learning for Financial Time Series Prediction." *Computers, Materials & Continua*, 139(1), 193–230.

[3] Taylor, S.J. (2007). *Modelling Financial Time Series* (2nd ed.). World Scientific.

[4] Ke, Z., et al. (2025). "A comprehensive survey of deep learning for time series forecasting: architectural diversity and open challenges." *Artificial Intelligence Review*, 58, Article 175.

[5] Li, W. & Law, K.E. (2024). "Deep learning models for time series forecasting: A review." *IEEE Access*. DOI: 10.1109/ACCESS.2024.3422528.

[6] Fischer, T. & Krauss, C. (2018). "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654–669.

[7] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." *arXiv preprint arXiv:1412.3555*.

[8] Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems (NIPS) 30*, pp. 3149–3157.

[9] Ju, Y., et al. (2020). "An improved Stacking framework for stock index prediction by leveraging tree-based ensemble models and deep learning algorithms." *Physica A*, 541, 122272.

[10] Liu, Y., et al. (2024). "Development of a Time Series E-Commerce Sales Prediction Method for Short-Shelf-Life Products Using GRU-LightGBM." *Applied Sciences*, 14(2), 866.

[11] World Gold Council (2024). *Gold Demand Trends Full Year 2024*. https://www.gold.org/

[12] Amini, A. & Kalantari, R. (2024). "Gold price prediction by a CNN-Bi-LSTM model along with automatic parameter tuning." *PLOS ONE*, 19(3), e0298426.

[13] López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. ISBN: 978-1-119-48208-6.

[14] Bailey, D.H., Borwein, J.M., López de Prado, M., & Zhu, Q.J. (2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance*, 20(4), 458–471.
