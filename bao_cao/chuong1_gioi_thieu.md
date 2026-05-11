# Chương 1. Giới thiệu

## 1.1. Bối cảnh nghiên cứu

Thị trường tài chính là môi trường dữ liệu khó cho học máy vì giá tài sản chịu ảnh hưởng đồng thời của thông tin vĩ mô, tâm lý thị trường, dòng tiền, thanh khoản và các sự kiện bất thường. Lý thuyết thị trường hiệu quả cho rằng giá đã phản ánh nhanh thông tin sẵn có, vì vậy việc tìm tín hiệu dự báo ổn định là thách thức lớn [1]. Thực nghiệm về đặc tính thống kê của lợi suất tài sản cũng cho thấy dữ liệu tài chính thường có phân phối đuôi dày, cụm biến động, bất đối xứng và thay đổi chế độ theo thời gian [2]. Những đặc điểm này khiến mô hình học máy dễ bị overfit nếu đánh giá không đúng.

Với XAU/USD, vàng vừa là tài sản hàng hóa, vừa là tài sản trú ẩn và công cụ dự trữ. Nhu cầu vàng chịu ảnh hưởng bởi lãi suất, lạm phát, rủi ro địa chính trị, nhu cầu ngân hàng trung ương và dòng tiền đầu tư [3]. Vì vậy, dự báo tín hiệu giao dịch XAU/USD không nên được trình bày như một bài toán “dự đoán chắc chắn giá tăng/giảm”, mà nên được xem như bài toán xây dựng một quy trình học máy có kiểm soát, minh bạch và tránh rò rỉ dữ liệu.

Đề tài này tập trung vào khung thời gian H1. Mỗi mẫu dữ liệu đại diện cho một thanh giá 1 giờ, từ đó hệ thống tạo đặc trưng kỹ thuật, gán nhãn bằng phương pháp triple-barrier, huấn luyện mô hình và đánh giá bằng walk-forward validation. Mục tiêu không phải chứng minh chiến lược CFD vàng sinh lời ổn định, mà là chứng minh một pipeline nghiên cứu có đủ các thành phần học thuật cần thiết: dữ liệu sạch, feature causal, nhãn có ý nghĩa giao dịch, validation theo thời gian, baseline comparison và báo cáo kết quả trung thực.

## 1.2. Vấn đề nghiên cứu

Bài toán được mô hình hóa thành phân loại ba lớp:

```text
Short (-1): ưu tiên tín hiệu bán
Hold  (0): không đủ điều kiện giao dịch
Long  (+1): ưu tiên tín hiệu mua
```

Khác với dự báo hồi quy giá tuyệt đối, phân loại tín hiệu phù hợp hơn với mục tiêu ra quyết định. Tuy nhiên, cách gán nhãn “giá sau N nến cao hơn hiện tại thì Long, thấp hơn thì Short” thường quá đơn giản vì không xét stop-loss, take-profit, thời gian giữ lệnh và biến động thị trường. Để khắc phục, đề tài dùng triple-barrier labeling theo hướng của López de Prado [5], trong đó mỗi mẫu có take-profit, stop-loss và horizon.

## 1.3. Mục tiêu đề tài

Đồ án xây dựng pipeline dự báo tín hiệu XAU/USD H1 với các mục tiêu cụ thể:

1. Chuẩn hóa dữ liệu tick/OHLCV thành bộ dữ liệu H1 có kiểm tra chất lượng.
2. Sinh đặc trưng kỹ thuật theo hướng causal, chỉ sử dụng thông tin quá khứ và hiện tại.
3. Tạo nhãn Short/Hold/Long bằng triple-barrier labeling dựa trên ATR.
4. Huấn luyện và đánh giá mô hình bằng walk-forward validation có purge/embargo nhằm giảm leakage.
5. So sánh baseline và các mô hình: Naive/Majority/Random baseline, Logistic Regression, Random Forest, LightGBM và Classic Hybrid Stacking.
6. Đánh giá bằng Accuracy, Balanced Accuracy, Directional Accuracy, Macro F1, per-class F1 và confusion matrix.
7. Dùng backtest như minh họa ứng dụng, không dùng làm bằng chứng chính cho hiệu quả mô hình.

## 1.4. Kiến trúc đề xuất

Kiến trúc runtime chính là Classic Hybrid Stacking:

```text
Base learners: Logistic Regression + Random Forest + LightGBM
Meta learner : Logistic Regression
Output       : Short / Hold / Long
```

Hybrid trong đề tài không có nghĩa là kết hợp mạng sâu với mô hình cây, mà là kết hợp nhiều họ mô hình có bản chất khác nhau. Logistic Regression đại diện cho mô hình tuyến tính dễ giải thích; Random Forest đại diện cho bagging tree có khả năng giảm phương sai [15]; LightGBM đại diện cho gradient boosting tree hiệu quả trên dữ liệu tabular [16], [17]. Meta-model Logistic Regression học cách kết hợp xác suất dự báo của các base learners theo nguyên lý stacked generalization [18].

## 1.5. Đóng góp chính

Đóng góp chính của đồ án là quy trình đánh giá minh bạch cho bài toán học máy tài chính:

- Không dùng random split cho chuỗi thời gian.
- Dùng walk-forward validation để mô phỏng huấn luyện trên quá khứ và kiểm tra trên tương lai.
- Dùng purge/embargo để giảm rò rỉ thông tin do nhãn có horizon [5].
- Dùng triple-barrier labeling để nhãn phản ánh logic TP/SL/horizon.
- So sánh mô hình với baseline thay vì chỉ báo cáo một mô hình duy nhất.
- Trình bày trung thực khi mô hình phức tạp không vượt mô hình đơn lẻ.

## 1.6. Phạm vi và giới hạn

Đề tài chỉ sử dụng dữ liệu giá và các đặc trưng kỹ thuật từ OHLCV, chưa tích hợp tin tức, dữ liệu vĩ mô, positioning, order book hoặc sentiment. Kết quả backtest là mô phỏng đơn giản với giả định về chi phí và lot size, không đại diện cho hiệu quả triển khai thực tế. Các kết luận trong đồ án nên được hiểu là kết quả nghiên cứu pipeline và phương pháp đánh giá, không phải khuyến nghị đầu tư.

## 1.7. Câu hỏi nghiên cứu

Để tránh việc đề tài bị hiểu như một hệ thống giao dịch hoàn chỉnh, các câu hỏi nghiên cứu được đặt theo hướng kiểm định pipeline học máy:

1. Có thể xây dựng một pipeline từ dữ liệu XAU/USD H1 đến dự báo Short/Hold/Long mà không dùng thông tin tương lai hay không?
2. Triple-barrier labeling dựa trên ATR có tạo được nhãn có ý nghĩa giao dịch hơn nhãn tăng/giảm đơn giản hay không?
3. Walk-forward validation có purge/embargo có giúp đánh giá thực tế hơn random split trong bối cảnh nhãn có event horizon hay không?
4. Classic Hybrid Stacking có cải thiện so với Logistic Regression, Random Forest và LightGBM đơn lẻ hay không?
5. Khi mô hình phức tạp không vượt baseline mạnh, có thể rút ra kết luận học thuật gì về dữ liệu tài chính nhiễu cao?
6. Backtest demo cho thấy tín hiệu có thể chuyển thành hành động giao dịch như thế nào và còn thiếu gì để tiến tới triển khai thực tế?

Những câu hỏi này giúp luận văn có cấu trúc kiểm định rõ ràng. Nếu chỉ hỏi “mô hình có kiếm tiền không”, báo cáo sẽ phụ thuộc quá nhiều vào một kết quả backtest cụ thể, trong khi backtest có thể bị ảnh hưởng mạnh bởi chi phí giao dịch, thời điểm chọn mẫu, tham số threshold và các giả định thực thi.

## 1.8. Động cơ chọn XAU/USD H1

XAU/USD được chọn vì vàng là tài sản có vai trò đặc biệt trong tài chính toàn cầu. Vàng thường được xem như tài sản trú ẩn khi rủi ro thị trường tăng, đồng thời chịu tác động bởi lãi suất thực, sức mạnh USD, lạm phát kỳ vọng và nhu cầu của ngân hàng trung ương [3]. Do đó, giá vàng có nhiều giai đoạn biến động mạnh và thay đổi regime, phù hợp để kiểm tra độ bền của pipeline học máy.

Khung H1 được chọn vì cân bằng giữa số lượng mẫu và mức độ nhiễu:

| Khung | Ưu điểm | Nhược điểm |
|---|---|---|
| M1/M5 | Nhiều mẫu, phản ứng nhanh | Rất nhạy với spread, slippage, microstructure noise |
| M15/M30 | Cân bằng hơn intraday | Vẫn có nhiễu ngắn hạn cao |
| H1 | Đủ mẫu, ít nhiễu hơn, phù hợp horizon 24h | Có thể bỏ lỡ tín hiệu rất ngắn hạn |
| H4/D1 | Ít nhiễu hơn | Số mẫu ít, khó train walk-forward nhiều cửa sổ |

Vì mục tiêu là đồ án học máy có kiểm soát, H1 là lựa chọn hợp lý hơn các khung quá nhỏ. Nó cho phép tạo đủ cửa sổ walk-forward nhưng vẫn tránh phần lớn nhiễu microstructure.

## 1.9. Đối tượng và phạm vi nghiên cứu

Đối tượng nghiên cứu là pipeline dự báo tín hiệu giao dịch dựa trên dữ liệu OHLCV của XAU/USD. Phạm vi bao gồm:

- Tiền xử lý dữ liệu OHLCV H1.
- Feature engineering kỹ thuật causal.
- Gán nhãn Short/Hold/Long bằng triple-barrier.
- Huấn luyện mô hình học máy cổ điển.
- Đánh giá bằng walk-forward validation.
- Minh họa tín hiệu bằng backtest đơn giản.

Phạm vi không bao gồm:

- Tối ưu hệ thống giao dịch thực chiến.
- Dự báo tin tức hoặc dữ liệu macro theo thời gian thực.
- Quản trị vốn nâng cao ở cấp portfolio.
- Kết nối broker hoặc giao dịch tự động.
- Cam kết lợi nhuận trong thực tế.

## 1.10. Ý nghĩa khoa học và thực tiễn

Về khoa học, đề tài minh họa cách áp dụng các khuyến nghị của financial machine learning vào một bài toán cụ thể: tránh random split, dùng nhãn event-based, kiểm soát leakage, so sánh baseline và phân tích per-class. Đây là các điểm thường bị bỏ qua trong các bài toán dự báo tài chính đơn giản.

Về thực tiễn, pipeline tạo ra một khung thử nghiệm có thể mở rộng. Khi có dữ liệu mới hoặc giả thuyết mới, người nghiên cứu có thể thay đổi label, feature hoặc mô hình nhưng vẫn giữ nguyên quy trình đánh giá. Điều này quan trọng hơn một kết quả backtest đơn lẻ vì nó giúp tránh việc “chọn tham số theo kết quả”.

## 1.11. Cấu trúc luận văn

Luận văn được tổ chức như sau:

- Chương 1 giới thiệu bối cảnh, mục tiêu, phạm vi và đóng góp.
- Chương 2 trình bày cơ sở lý thuyết về dữ liệu tài chính, labeling, validation, mô hình và metrics.
- Chương 3 trình bày dữ liệu, tiền xử lý, feature engineering và label distribution.
- Chương 4 trình bày phương pháp đề xuất và thiết kế pipeline.
- Chương 5 trình bày thực nghiệm và phân tích kết quả.
- Chương 6 minh họa cách chuyển tín hiệu mô hình thành backtest demo.
- Chương 7 kết luận, nêu hạn chế và hướng phát triển.
