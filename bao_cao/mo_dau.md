# MỞ ĐẦU

## 1. Tính cấp thiết của đề tài

Thị trường tài chính là môi trường dữ liệu phức tạp cho học máy vì giá tài sản chịu ảnh hưởng đồng thời của thông tin vĩ mô, tâm lý thị trường, dòng tiền, thanh khoản và các sự kiện bất thường. Lý thuyết thị trường hiệu quả cho rằng giá đã phản ánh nhanh thông tin sẵn có, do đó việc tìm tín hiệu dự báo ổn định là thách thức lớn [1]. Thực nghiệm về đặc tính thống kê của lợi suất tài sản cũng cho thấy dữ liệu tài chính thường có phân phối đuôi dày, cụm biến động, bất đối xứng và thay đổi chế độ theo thời gian [2]. Những đặc điểm này khiến mô hình học máy dễ bị overfit nếu đánh giá không đúng.

Vàng (XAU/USD) là tài sản có vai trò đặc biệt trong tài chính toàn cầu. Vàng vừa là tài sản hàng hóa, vừa là tài sản trú ẩn và công cụ dự trữ. Nhu cầu vàng chịu ảnh hưởng bởi lãi suất, lạm phát, rủi ro địa chính trị, nhu cầu ngân hàng trung ương và dòng tiền đầu tư [3]. Trong bối cảnh biến động kinh tế toàn cầu, việc nghiên cứu quy trình dự báo tín hiệu giao dịch vàng có ý nghĩa cả về mặt khoa học lẫn thực tiễn.

Bên cạnh đó, việc áp dụng học máy vào tài chính đặt ra nhiều thách thức về phương pháp luận. Nhiều nghiên cứu dự báo giá tài sản sử dụng random split hoặc không kiểm soát rò rỉ dữ liệu, dẫn đến kết quả đánh giá quá lạc quan và không có giá trị thực tiễn [5], [7]. Đây là lý do để xây dựng một pipeline nghiên cứu có kiểm soát, minh bạch và tránh rò rỉ dữ liệu.

## 2. Lý do chọn đề tài

XAU/USD được chọn vì vàng là tài sản có thanh khoản cao và chịu tác động của nhiều yếu tố kinh tế vĩ mô. Vàng thường được xem như tài sản trú ẩn khi rủi ro thị trường tăng, đồng thời chịu tác động bởi lãi suất thực, sức mạnh USD, lạm phát kỳ vọng và nhu cầu của ngân hàng trung ương [3]. Do đó, giá vàng có nhiều giai đoạn biến động mạnh và thay đổi chế độ, phù hợp để kiểm tra độ bền của pipeline học máy.

Khung thời gian H1 được chọn vì cân bằng giữa số lượng mẫu và mức độ nhiễu. Khung nhỏ hơn như M1/M5 thường nhiễu hơn và nhạy với spread và slippage do microstructure; khung lớn hơn như D1 có ít mẫu hơn, khó train walk-forward với nhiều cửa sổ. H1 là lựa chọn hợp lý cho nghiên cứu có kiểm soát.

## 3. Mục tiêu nghiên cứu

Đồ án xây dựng pipeline dự báo tín hiệu XAU/USD H1 với các mục tiêu cụ thể:

1. Chuẩn hóa dữ liệu tick/OHLCV thành bộ dữ liệu H1 có kiểm tra chất lượng.
2. Sinh đặc trưng kỹ thuật theo hướng causal, chỉ sử dụng thông tin quá khứ và hiện tại.
3. Tạo nhãn Short/Hold/Long bằng triple-barrier labeling dựa trên ATR.
4. Huấn luyện và đánh giá mô hình bằng walk-forward validation có purge/embargo nhằm giảm leakage.
5. So sánh baseline và các mô hình: Naive/Majority/Random baseline, Logistic Regression, Random Forest, LightGBM và Classic Hybrid Stacking.
6. Đánh giá bằng Accuracy, Balanced Accuracy, Directional Accuracy, Macro F1, per-class F1 và confusion matrix.
7. Dùng backtest như minh họa ứng dụng, không dùng làm bằng chứng chính cho hiệu quả mô hình.

## 4. Đối tượng và phạm vi nghiên cứu

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

## 5. Phương pháp nghiên cứu

Đề tài sử dụng các phương pháp nghiên cứu sau:

- Nghiên cứu tài liệu về thị trường tài chính, đặc tính dữ liệu tài chính và học máy tài chính.
- Thiết kế pipeline theo phương pháp stacked generalization kết hợp nhiều mô hình học máy.
- Đánh giá bằng walk-forward validation có purge/embargo để kiểm soát rò rỉ dữ liệu.
- So sánh với nhiều baseline để đảm bảo kết quả có giá trị tham chiếu.
- Phân tích kết quả dựa trên metrics phù hợp cho bài toán phân loại ba lớp có mất cân bằng.

## 6. Ý nghĩa khoa học và thực tiễn

Về khoa học, đề tài minh họa cách áp dụng các khuyến nghị của financial machine learning vào một bài toán cụ thể: tránh random split, dùng nhãn event-based, kiểm soát leakage, so sánh baseline và phân tích per-class. Đây là các điểm thường bị bỏ qua trong các bài toán dự báo tài chính đơn giản.

Về thực tiễn, pipeline tạo ra một khung thử nghiệm có thể mở rộng. Khi có dữ liệu mới hoặc giả thuyết mới, người nghiên cứu có thể thay đổi label, feature hoặc mô hình nhưng vẫn giữ nguyên quy trình đánh giá. Điều này quan trọng hơn một kết quả backtest đơn lẻ vì nó giúp tránh việc "chọn tham số theo kết quả".

## 7. Cấu trúc luận văn

Luận văn được tổ chức theo cấu trúc chuẩn của báo cáo đồ án tốt nghiệp:

- **Chương 1: Tổng quan** trình bày bối cảnh, động cơ chọn XAU/USD, bài toán nghiên cứu, câu hỏi nghiên cứu, mục tiêu, phạm vi, đóng góp và cấu trúc luận văn.
- **Chương 2: Cơ sở lý thuyết** trình bày nền tảng học thuật: đặc tính dữ liệu tài chính, phân tích kỹ thuật dưới góc nhìn thống kê, triple-barrier labeling, walk-forward validation, purge/embargo, các mô hình học máy, metrics, SHAP và rủi ro backtest overfitting.
- **Chương 3: Dữ liệu và phương pháp** trình bày dữ liệu và pipeline: nguồn, biến OHLCV, múi giờ, kiểm tra chất lượng, feature engineering causal, whitelist feature, gán nhãn, phân phối nhãn và thiết kế pipeline 6 stage.
- **Chương 4: Thực nghiệm và kết quả** trình bày kết quả: môi trường, cấu hình, so sánh mô hình, metrics, confusion matrix, high-confidence, backtest demo, phân tích lỗi và hạn chế.
- **Chương 5: Minh họa ứng dụng** trình bày cách chuyển tín hiệu dự báo thành hành động giao dịch, quy tắc thực thi, kết quả backtest demo và điều kiện triển khai thực tế.
- **Chương 6: Kết luận và kiến nghị** tóm tắt đóng góp, kết quả thực nghiệm, hạn chế và hướng phát triển.
- **Tài liệu tham khảo** liệt kê các nguồn học thuật được sử dụng trong luận văn.

Nguyên tắc viết:

- Không viết theo hướng "mô hình chắc chắn kiếm tiền".
- Viết theo hướng "pipeline đánh giá học máy tài chính có kiểm soát".
- Kết quả không thắng baseline vẫn có giá trị nếu được phân tích trung thực.
- Mọi claim học thuật nên gắn với tài liệu tham khảo.
- Mọi claim thực nghiệm nên gắn với artifact trong thư mục results.

## 8. Giới hạn của đề tài

Đề tài cần được hiểu trong bối cảnh một số giới hạn:

- Chỉ sử dụng dữ liệu giá và các đặc trưng kỹ thuật từ OHLCV, chưa tích hợp tin tức, dữ liệu vĩ mô, positioning, order book hoặc sentiment.
- Kết quả backtest là mô phỏng đơn giản với giả định về chi phí và lot size, không đại diện cho hiệu quả triển khai thực tế.
- Các kết luận trong đồ án nên được hiểu là kết quả nghiên cứu pipeline và phương pháp đánh giá, không phải khuyến nghị đầu tư.
