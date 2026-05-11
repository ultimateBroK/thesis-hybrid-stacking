# Chương 2. Cơ sở lý thuyết

## 2.1. Đặc điểm dữ liệu tài chính

Dữ liệu tài chính khác với nhiều bộ dữ liệu supervised learning thông thường ở ba điểm quan trọng. Thứ nhất, dữ liệu có thứ tự thời gian và phân phối có thể thay đổi theo thời gian. Thứ hai, tín hiệu dự báo thường yếu so với nhiễu. Thứ ba, việc thử nhiều chiến lược/mô hình trên cùng dữ liệu dễ tạo ra kết quả đẹp nhưng không bền vững ngoài mẫu.

Cont tổng hợp nhiều “stylized facts” của lợi suất tài sản như đuôi dày, volatility clustering và tính không chuẩn của phân phối lợi suất [2]. Các đặc điểm này khiến giả định IID thường không phù hợp. Trong tài chính, nếu dùng cross-validation ngẫu nhiên, mô hình có thể học thông tin từ tương lai hoặc từ các mẫu có nhãn chồng lấn với test set, dẫn đến đánh giá quá lạc quan [5], [7].

## 2.2. Bài toán phân loại tín hiệu giao dịch

Đề tài mô hình hóa bài toán thành phân loại ba lớp:

- Short (-1): ưu tiên bán hoặc tránh vị thế mua.
- Hold (0): không đủ tín hiệu vào lệnh.
- Long (+1): ưu tiên mua hoặc tránh vị thế bán.

Phân loại ba lớp phù hợp hơn hồi quy giá đóng cửa vì quyết định giao dịch thường không cần dự báo giá chính xác từng điểm, mà cần xác định hướng và điều kiện có đủ lợi thế sau chi phí giao dịch hay không. Lớp Hold có vai trò quan trọng vì trong nhiều thời điểm thị trường không có tín hiệu đủ mạnh; ép mọi mẫu thành Long/Short có thể làm mô hình giao dịch quá nhiều.

## 2.3. Feature engineering kỹ thuật

Phân tích kỹ thuật được dùng rộng rãi trong giao dịch để mô tả xu hướng, dao động, biến động và vị trí giá. Lo, Mamaysky và Wang cho thấy có thể nghiên cứu các mẫu hình kỹ thuật bằng phương pháp thống kê thay vì chỉ dựa vào kinh nghiệm chủ quan [11]. Trong đồ án, các chỉ báo không được xem là “quy tắc giao dịch” độc lập, mà là đặc trưng đầu vào cho mô hình học máy.

Các nhóm đặc trưng gồm:

1. Trend: EMA distance, EMA slope, ADX.
2. Momentum: returns nhiều chu kỳ, RSI, MACD histogram.
3. Volatility: ATR normalized, ATR ratio, high-low range.
4. Price position: vị trí giá trong range, pivot position, VWAP.
5. Session: phiên Á, London, New York morning/afternoon.

Nguyên tắc quan trọng là causal: tại thời điểm t, đặc trưng chỉ được tính từ dữ liệu không muộn hơn t. Các cột như label, barrier, event_end, giá tương lai hoặc metadata gán nhãn không được đưa vào tập feature.

## 2.4. Triple-barrier labeling

Triple-barrier labeling gán nhãn dựa trên ba điều kiện [5]:

1. Upper barrier: giá chạm ngưỡng take-profit trước.
2. Lower barrier: giá chạm ngưỡng stop-loss trước.
3. Vertical barrier: hết horizon mà chưa chạm TP/SL.

Với mỗi thời điểm t, hệ thống đặt TP và SL theo ATR:

```text
upper_barrier = close_t + ATR_t * atr_tp_multiplier
lower_barrier = close_t - ATR_t * atr_sl_multiplier
vertical_barrier = t + horizon_bars
```

Nếu upper barrier chạm trước lower barrier, nhãn là Long. Nếu lower barrier chạm trước upper barrier, nhãn là Short. Nếu không barrier nào chạm trước horizon, nhãn là Hold. Cách này tốt hơn nhãn tăng/giảm sau N nến vì nó gắn nhãn với logic giao dịch có rủi ro và thời hạn.

## 2.5. Walk-forward validation

Walk-forward validation là phương pháp đánh giá theo thứ tự thời gian: huấn luyện trên một đoạn quá khứ, kiểm tra trên đoạn tương lai kế tiếp, sau đó trượt cửa sổ và lặp lại. Cách này gần hơn với triển khai thực tế so với random split vì tại thời điểm dự báo, mô hình chỉ được học từ dữ liệu quá khứ.

Trong tài chính, walk-forward còn giúp quan sát tính ổn định của mô hình qua nhiều giai đoạn thị trường. Nếu mô hình chỉ tốt ở một vài cửa sổ nhưng kém ở phần còn lại, kết quả tổng thể cần được diễn giải thận trọng.

## 2.6. Purge và embargo

Với nhãn triple-barrier, một mẫu tại thời điểm t có thể sử dụng thông tin giá đến t + horizon. Nếu một mẫu train có khoảng sự kiện chồng lấn với test set, mô hình có thể gián tiếp nhìn thấy thông tin tương lai. López de Prado đề xuất purged cross-validation và embargo để loại bỏ hoặc cách ly các mẫu train có khả năng overlap với test event [5].

Trong pipeline này:

- Purge loại bỏ các mẫu train có event_end chồng với giai đoạn test.
- Embargo tạo khoảng đệm sau test window để giảm ảnh hưởng lan truyền thông tin.
- Nếu không có event_end, hệ thống fallback sang purge cố định theo số bar.

## 2.7. Các mô hình học máy sử dụng

### 2.7.1. Logistic Regression

Logistic Regression là mô hình tuyến tính cho phân loại xác suất. Ưu điểm là đơn giản, ít tham số, dễ giải thích và thường là baseline mạnh cho dữ liệu nhiều nhiễu [14]. Trong đồ án, Logistic Regression được dùng vừa làm base learner vừa làm meta learner.

### 2.7.2. Random Forest

Random Forest kết hợp nhiều decision trees huấn luyện trên bootstrap samples và chọn ngẫu nhiên một phần feature ở mỗi split [15]. Cơ chế bagging giúp giảm phương sai so với một cây đơn lẻ. Trong tài chính, Random Forest thường hữu ích vì có thể học quan hệ phi tuyến và tương tác giữa feature mà không cần giả định tuyến tính.

### 2.7.3. LightGBM

LightGBM là triển khai gradient boosting decision tree có hiệu quả cao, sử dụng các kỹ thuật như Gradient-based One-Side Sampling (GOSS) và Exclusive Feature Bundling (EFB) để tăng tốc huấn luyện [17]. Gradient boosting nói chung xây dựng mô hình cộng dồn nhiều cây yếu nhằm tối ưu hàm mất mát [16]. Với dữ liệu tabular có nhiều feature kỹ thuật, LightGBM thường là baseline mạnh.

### 2.7.4. Classic Hybrid Stacking

Stacked generalization là phương pháp học cách kết hợp đầu ra của nhiều mô hình tầng dưới bằng một mô hình tầng trên [8]. Trong đồ án, mỗi base learner xuất xác suất cho ba lớp Short/Hold/Long. Các xác suất này được ghép thành meta-features và đưa vào Logistic Regression meta-model.

Điểm quan trọng là split meta phải theo thời gian. Nếu dùng cùng dữ liệu để vừa train base model vừa train meta model trên dự báo in-sample, meta learner sẽ học dự báo quá lạc quan. Vì vậy pipeline chia train window thành base-train và meta-train theo thứ tự thời gian.

## 2.8. Chỉ số đánh giá

Các chỉ số chính:

- Accuracy: tỷ lệ dự báo đúng toàn bộ ba lớp.
- Balanced Accuracy: trung bình recall theo lớp, hữu ích khi mất cân bằng lớp.
- Macro F1: trung bình F1 của từng lớp, không trọng số theo số lượng mẫu.
- Per-class F1: cho biết lớp nào bị mô hình bỏ qua.
- Directional Accuracy: chỉ xét các mẫu có hướng Long/Short, phục vụ diễn giải giao dịch.
- Confusion matrix: phân tích nhầm lẫn giữa Short/Hold/Long.

Macro F1 được ưu tiên hơn accuracy khi phân phối lớp lệch vì accuracy có thể cao nếu mô hình chỉ dự báo lớp đa số. Cách tính precision, recall, F1 và macro average tương ứng với chuẩn đánh giá phân loại phổ biến trong scikit-learn [21].

## 2.9. Diễn giải mô hình và feature importance

Với mô hình cây như LightGBM, feature importance giúp xác định đặc trưng nào đóng góp nhiều vào split hoặc gain. Tuy nhiên, importance nội bộ của cây có thể bị ảnh hưởng bởi tương quan feature. SHAP cung cấp cách diễn giải thống nhất dựa trên giá trị Shapley, gán đóng góp cho từng feature ở từng dự báo [22]. Trong phạm vi đồ án, feature importance được dùng để giảm feature nhiễu; SHAP được đề xuất như hướng phát triển để phân tích sâu hơn theo regime thị trường.

## 2.10. Rủi ro backtest overfitting

Backtest có thể gây ảo tưởng hiệu quả nếu nhà nghiên cứu thử nhiều tham số, nhiều mô hình và chỉ chọn kết quả tốt nhất. Bailey và cộng sự chỉ ra xác suất overfit tăng khi số lượng thử nghiệm tăng [7]. Vì vậy, backtest trong đồ án chỉ được xem là minh họa ứng dụng tín hiệu. Bằng chứng chính của mô hình là kết quả classification theo walk-forward validation, có baseline và báo cáo per-class.

## 2.11. Tính không dừng và thay đổi chế độ thị trường

Một giả định quan trọng của nhiều thuật toán học máy là phân phối train và test tương đối giống nhau. Trong tài chính, giả định này thường bị vi phạm. Thị trường có thể chuyển từ giai đoạn trend sang sideway, từ biến động thấp sang biến động cao hoặc từ môi trường lãi suất thấp sang lãi suất cao. Khi phân phối thay đổi, mô hình học từ quá khứ có thể không còn phù hợp với tương lai.

Đối với XAU/USD, thay đổi regime có thể đến từ:

- Chính sách tiền tệ của Cục Dự trữ Liên bang Mỹ.
- Kỳ vọng lạm phát và lãi suất thực.
- Rủi ro địa chính trị.
- Sức mạnh hoặc suy yếu của USD.
- Nhu cầu vàng vật chất và dự trữ ngân hàng trung ương.
- Sự kiện thanh khoản bất thường trong phiên Mỹ hoặc London.

Vì vậy, báo cáo không nên chỉ nhìn metric tổng. Cần xem thêm kết quả theo cửa sổ thời gian, theo giai đoạn thị trường và theo class. Một mô hình có accuracy trung bình chấp nhận được nhưng sụp đổ ở một vài regime vẫn có rủi ro lớn khi triển khai.

## 2.12. Vì sao random split không phù hợp

Random split giả định các mẫu có thể hoán đổi vị trí và độc lập tương đối. Với chuỗi thời gian tài chính, điều này sai ở hai cấp:

1. Cấp thời gian: mẫu test có thể xảy ra trước mẫu train nếu xáo trộn ngẫu nhiên, tương đương cho mô hình học từ tương lai.
2. Cấp event label: với triple-barrier, nhãn của mẫu tại t phụ thuộc vào diễn biến giá đến t + horizon. Hai mẫu gần nhau có thể dùng chung đoạn giá tương lai để xác định nhãn.

Ví dụ, nếu mẫu test bắt đầu tại 10:00 và mẫu train bắt đầu tại 09:00 nhưng có horizon 24 giờ, nhãn train có thể dùng giá đến ngày hôm sau, chồng lấn với vùng thông tin của test. Đây là lý do cần purge/embargo [5].

## 2.13. Mối quan hệ giữa TP/SL, horizon và nhãn Hold

Trong triple-barrier, tỷ lệ lớp Hold phụ thuộc vào ba yếu tố:

- Độ rộng TP/SL.
- Độ dài horizon.
- Mức biến động thị trường.

Nếu TP/SL quá hẹp, giá dễ chạm barrier, lớp Hold thấp. Nếu TP/SL quá rộng hoặc horizon quá ngắn, nhiều mẫu không chạm barrier, lớp Hold cao. Nếu horizon dài hơn, khả năng chạm barrier thường tăng, do đó Hold có thể giảm. Đây chính là hiện tượng quan sát được trong thực nghiệm: horizon 48 làm Hold giảm còn khoảng 1.5%.

Vì vậy, label design không thể tách rời dữ liệu. Một cấu hình có vẻ hợp lý trên lý thuyết vẫn phải được kiểm tra bằng phân phối nhãn thực tế.

## 2.14. Bias-variance trong lựa chọn mô hình

Logistic Regression có bias cao hơn nhưng variance thấp hơn; mô hình ít phức tạp nên khó overfit hơn. Random Forest giảm variance bằng bagging nhưng có thể kém trong việc ngoại suy xu hướng. Gradient boosting như LightGBM có khả năng học phi tuyến mạnh nhưng cần regularization để tránh học nhiễu [16], [17].

Stacking thêm một tầng học nữa. Nếu các base learners có lỗi khác nhau, stacking có thể học cách kết hợp để cải thiện. Nhưng nếu tất cả base learners đều bị giới hạn bởi tín hiệu yếu, lỗi tương quan hoặc label noise, meta-model có thể không cải thiện. Điều này giải thích vì sao Hybrid Stacking không nhất thiết vượt LightGBM trong mọi lần chạy.

## 2.15. Baseline trong học máy tài chính

Baseline là bắt buộc vì thị trường có thể có class imbalance hoặc persistence tự nhiên. Nếu không có baseline, một accuracy nhìn có vẻ tốt có thể thực ra chỉ là mô hình học lớp đa số.

Các baseline có ý nghĩa:

| Baseline | Ý nghĩa |
|---|---|
| Majority Class | Kiểm tra mô hình có vượt việc dự báo lớp phổ biến nhất không |
| Random Baseline | Floor hiệu suất ngẫu nhiên |
| Naive Direction | Kiểm tra persistence hướng giá |
| Logistic Regression | Baseline học máy tuyến tính |
| LightGBM | Baseline tabular mạnh |

Trong báo cáo, việc LightGBM vượt Hybrid Stacking là thông tin quan trọng. Nó cho thấy mô hình đơn lẻ có thể phù hợp hơn trong điều kiện dữ liệu hiện tại.

## 2.16. Macro F1 và bài toán mất cân bằng lớp

Với ba lớp Short/Hold/Long, nếu Hold chỉ chiếm khoảng 9%, mô hình có thể đạt accuracy tương đối bằng cách tập trung vào Short/Long. Macro F1 khắc phục một phần vấn đề này vì mỗi lớp có trọng số ngang nhau. Nếu F1 của Hold thấp, Macro F1 sẽ phản ánh điều đó.

Công thức:

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
Macro F1    = mean(F1_short, F1_hold, F1_long)
```

Macro F1 không thay thế hoàn toàn phân tích confusion matrix. Nó chỉ là số tổng hợp. Báo cáo vẫn cần xem mô hình nhầm Short thành Long, Long thành Short hay đẩy nhiều mẫu sang Hold.

## 2.17. Xác suất dự báo và calibration

Nhiều mô hình phân loại xuất xác suất, nhưng xác suất này không luôn được calibration tốt. Một mô hình dự báo xác suất Long 0.70 không có nghĩa trong thực tế 70% các mẫu như vậy sẽ Long. Calibration kiểm tra sự khớp giữa xác suất dự báo và tần suất đúng quan sát được [23].

Trong giao dịch, calibration quan trọng vì threshold vào lệnh thường dựa trên confidence. Nếu mô hình over-confident, hệ thống có thể vào lệnh quá nhiều. Nếu under-confident, hệ thống có thể bỏ lỡ cơ hội. Do đó, calibration là hướng phát triển tự nhiên sau khi pipeline classification ổn định.

## 2.18. Diễn giải mô hình bằng SHAP

Feature importance của LightGBM cho biết feature nào thường được dùng trong cây, nhưng không giải thích rõ từng dự báo cụ thể. SHAP dựa trên Shapley values để phân bổ đóng góp của từng feature cho một dự báo [22].

Trong bài toán này, SHAP có thể trả lời các câu hỏi:

- Khi mô hình dự báo Long, feature nào đẩy xác suất Long lên?
- Khi mô hình dự báo Hold sai, feature nào gây nhầm lẫn?
- Feature quan trọng có ổn định giữa các regime không?
- LightGBM và Hybrid Stacking có dựa vào cùng nhóm tín hiệu không?

Tuy nhiên SHAP tốn chi phí tính toán, nên trong phạm vi hiện tại chỉ nên xem là phân tích mở rộng hoặc phụ lục.

## 2.19. Kết luận cơ sở lý thuyết

Các lý thuyết trên dẫn đến quan điểm thiết kế của đồ án: mô hình không phải thành phần duy nhất quyết định chất lượng nghiên cứu. Với tài chính, cách gán nhãn, cách chia dữ liệu, cách chống leakage và cách so sánh baseline quan trọng không kém thuật toán. Một mô hình phức tạp hơn nhưng đánh giá sai sẽ kém giá trị hơn một mô hình đơn giản được kiểm định chặt chẽ.

## 2.20. Các dạng rò rỉ dữ liệu thường gặp trong bài toán này

Rò rỉ dữ liệu không chỉ xảy ra khi vô tình đưa cột nhãn vào mô hình. Trong bài toán tài chính có nhiều dạng leakage tinh vi hơn:

| Dạng leakage | Ví dụ | Cách kiểm soát trong đồ án |
|---|---|---|
| Target leakage | Đưa `label`, `touched_bar`, `upper_barrier` vào feature | EXCLUDE_COLS loại toàn bộ label metadata |
| Lookahead feature | Indicator dùng giá tương lai | Feature chỉ tính rolling/EMA từ quá khứ đến hiện tại |
| Random split leakage | Train chứa mẫu sau test | Walk-forward theo thời gian |
| Event overlap | Train label dùng đoạn giá trùng test | Event-time purge và embargo |
| Selection bias | Thử nhiều cấu hình rồi chọn kết quả đẹp | Ghi lại cấu hình và báo cáo kết quả trung thực |
| Backtest overfitting | Tối ưu rule giao dịch trên test set | Backtest chỉ là demo, không là bằng chứng chính |

Trong báo cáo, nên nhấn mạnh rằng kiểm soát leakage là một phần đóng góp chính. Nếu chỉ dùng một mô hình mạnh nhưng validation sai, kết quả có thể cao nhưng không có giá trị khoa học.

## 2.21. Average uniqueness và sample weighting

Trong event-based labeling, nhiều mẫu có thể chồng lấn thời gian sự kiện. Nếu nhiều nhãn cùng phụ thuộc vào cùng một đoạn giá tương lai, các mẫu đó không độc lập hoàn toàn. López de Prado đề xuất khái niệm uniqueness để đo mức độ độc lập tương đối của một mẫu [5].

Ý tưởng trực giác:

- Nếu tại một thời điểm chỉ có một event đang hoạt động, event đó có uniqueness cao.
- Nếu nhiều event cùng chồng lên một đoạn thời gian, mỗi event có uniqueness thấp hơn.
- Mẫu có uniqueness thấp nên có trọng số nhỏ hơn vì nó không mang nhiều thông tin độc lập.

Pipeline lưu `sample_weight` để Stage 4 có thể dùng trong training. Trọng số được chuẩn hóa về mean 1 để không làm thay đổi scale tổng thể của loss.

## 2.22. Vì sao dùng ATR cho barrier

ATR đo biến động tuyệt đối dựa trên high, low và close trước đó. Dùng ATR cho TP/SL có ba lợi ích:

1. Barrier thích nghi với biến động. Khi thị trường biến động mạnh, barrier rộng hơn; khi thị trường yên tĩnh, barrier hẹp hơn.
2. Barrier có ý nghĩa tương đối ổn định giữa các vùng giá khác nhau.
3. Label gắn với risk management thay vì khoảng giá cố định tùy ý.

Nếu dùng khoảng USD cố định, ví dụ 10 USD, ý nghĩa của barrier có thể thay đổi theo thời gian. Khi vàng ở 1200 USD, 10 USD là mức khác so với khi vàng ở 2500 USD. ATR giúp giảm vấn đề này.

## 2.23. Quan hệ giữa feature engineering và interpretability

Feature engineering không chỉ nhằm tăng hiệu suất. Với luận văn, feature có tên rõ ràng giúp giải thích mô hình:

- `ema34_vs_ema89` liên quan đến xu hướng trung hạn.
- `rsi_14` liên quan đến động lượng/quá mua quá bán.
- `atr_pct_close` liên quan đến biến động tương đối.
- `price_position_20` liên quan đến vị trí giá trong range gần đây.
- `sess_london`, `sess_ny_am` liên quan đến phiên giao dịch.

Nếu dùng raw sequence hoặc embedding khó hiểu, báo cáo sẽ khó bảo vệ hơn. Đây là lý do pipeline hiện tại ưu tiên classic tabular features và mô hình có thể giải thích.

## 2.24. Vì sao không lấy profitability làm metric chính

Profitability phụ thuộc vào nhiều yếu tố ngoài mô hình dự báo:

- Spread và commission.
- Slippage.
- Lot size.
- Leverage.
- Rule vào/thoát lệnh.
- Giới hạn số lệnh.
- Thời gian giữ lệnh.
- Điều kiện broker.

Hai mô hình có cùng dự báo classification có thể tạo kết quả backtest khác nhau nếu rule giao dịch khác nhau. Vì vậy, luận văn dùng classification metrics làm bằng chứng chính và profitability/backtest làm phần minh họa phụ.

## 2.25. Tổng kết mở rộng cơ sở lý thuyết

Cơ sở lý thuyết của đề tài có thể tóm tắt thành bốn lớp:

1. Lớp thị trường: dữ liệu tài chính nhiễu, non-stationary và khó dự báo.
2. Lớp dữ liệu: feature phải causal, nhãn phải có ý nghĩa giao dịch.
3. Lớp validation: train/test phải theo thời gian, có purge/embargo.
4. Lớp mô hình: mô hình phải được so sánh với baseline và giải thích bằng metrics phù hợp.

Bốn lớp này tạo nền tảng cho phương pháp ở Chương 4 và cách diễn giải kết quả ở Chương 5.


## 2.26. Ghi chú liên hệ với các chương sau

Các khái niệm trong chương này được dùng trực tiếp ở các chương sau: Chương 3 dùng nguyên tắc causal feature và ATR barrier; Chương 4 dùng walk-forward, purge/embargo và stacking; Chương 5 dùng Macro F1, confusion matrix và baseline comparison để diễn giải kết quả; Chương 6 dùng cảnh báo backtest overfitting để giới hạn phạm vi ứng dụng. Vì vậy, Chương 2 không chỉ là phần lý thuyết chung mà là nền tảng giải thích toàn bộ thiết kế thực nghiệm của luận văn.
