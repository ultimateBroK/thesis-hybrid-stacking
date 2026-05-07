# Chương 2. Cơ sở lý thuyết

## 2.1. Dự báo chuỗi thời gian tài chính

### 2.1.1. Đặc thù của chuỗi thời gian tài chính

Dữ liệu chuỗi thời gian tài chính (financial time series) có các đặc điểm khác biệt so với dữ liệu chuỗi thời gian trong các lĩnh vực khác:

- **Phi tuyến tính:** Mối quan hệ giữa các biến đầu vào và giá/khối lượng thường là phi tuyến, khó nắm bắt bằng các mô hình tuyến tính [1].
- **Phi tĩnh (non-stationarity):** Phân phối xác suất của chuỗi giá thay đổi theo thời gian do biến đổi chế độ thị trường (regime change) [2].
- **Nhiễu cao (high noise-to-signal ratio):** Tín hiệu dự báo thường yếu và bị che khuất bởi nhiễu ngẫu nhiên [3].
- **Tự tương quan yếu:** Khác với dữ liệu khí tượng hay sinh học, chuỗi giá tài chính thường có tự tương quan tuyến tính rất yếu ở các khung thời gian ngắn [4].
- **Phản ánh thông tin (information aggregation):** Thông tin thị trường được tích hợp vào giá theo cách phức tạp, phù hợp với giả thuyết thị trường hiệu quả (Efficient Market Hypothesis) ở mức độ yếu [5].

### 2.1.2. Phân loại (Classification) so với Hồi quy (Regression)

Bài toán dự báo tài chính có thể được tiếp cận theo hai hướng chính:

**Hồi quy (Regression):** Dự đoán giá trị liên tục của giá hoặc tỷ suất lợi nhuận trong tương lai. Ví dụ: dự đoán giá đóng cửa giờ tới là 2,650.50 USD/ounce.

**Phân loại (Classification):** Dự đoán hướng di chuyển của giá thành các lớp rời rạc. Trong đồ án này, sử dụng 3 lớp: Long (+1), Hold (0), Short (−1).

Lợi ích của hướng tiếp cận phân loại:

1. **Khớp với quyết định giao dịch:** Nhà giao dịch cần biết nên mua, bán, hay giữ — không cần giá trị tuyệt đối chính xác đến từng cent [6].
2. **Chịu lỗi tốt hơn:** Sai lệch 0.1% trong giá dự báo có thể dẫn đến sai hướng, nhưng phân loại trực tiếp tối ưu hóa cho đúng hướng [6].
3. **Giảm ảnh hưởng của giá trị ngoại lai:** Các sự kiện flash crash hay spike không ảnh hưởng nghiêm trọng đến nhãn lớp.

### 2.1.3. Phương pháp dự báo chuỗi thời gian

Các phương pháp dự báo chuỗi thời gian tài chính có thể được phân loại thành ba nhóm chính:

**Nhóm thống kê truyền thống:** ARIMA (Autoregressive Integrated Moving Average), GARCH (Generalized Autoregressive Conditional Heteroskedasticity), và VAR (Vector Autoregression) [7]. Các mô hình này có ưu điểm là khả năng diễn giải tốt, nhưng giả định tuyến tính và phân phối chuẩn.

**Nhóm học máy cổ điển (classical ML):** Support Vector Machine (SVM), Random Forest, Gradient Boosted Decision Trees [8]. Các phương pháp này xử lý phi tuyến tốt hơn nhưng không nắm bắt trực tiếp phụ thuộc thời gian.

**Nhóm học sâu (deep learning):** RNN, LSTM, GRU, Transformer [1, 9]. Các mô hình này có khả năng học các mẫu hình phức tạp và phụ thuộc thời gian dài hạn, nhưng đòi hỏi nhiều dữ liệu và tài nguyên tính toán.

## 2.2. Mạng Gated Recurrent Unit (GRU)

### 2.2.1. Kiến trúc GRU

GRU (Gated Recurrent Unit) được đề xuất bởi Cho et al. (2014) [10] như một biến thể đơn giản hơn của LSTM (Long Short-Term Memory) [11]. GRU sử dụng hai cổng (gate) thay vì ba như LSTM: cổng cập nhật (update gate) và cổng đặt lại (reset gate).

Cho một bước thời gian $t$, GRU tính toán như sau:

**Cổng đặt lại (reset gate):**

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

**Cổng cập nhật (update gate):**

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

**Ứng viên trạng thái ẩn (candidate hidden state):**

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

**Trạng thái ẩn mới:**

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

trong đó $\sigma$ là hàm sigmoid, $\odot$ là phép nhân từng phần tử (element-wise multiplication), $x_t$ là đầu vào tại thời điểm $t$, và $h_{t-1}$ là trạng thái ẩn trước đó.

### 2.2.2. Ưu điểm của GRU so với LSTM

- **Ít tham số hơn:** GRU có 2 cổng (update, reset) so với 3 cổng của LSTM (input, forget, output), giảm khoảng 25–30% số tham số huấn luyện [10].
- **Huấn luyện nhanh hơn:** Ít tham số đồng nghĩa với thời gian huấn luyện mỗi epoch ngắn hơn và giảm nguy cơ overfitting trên tập dữ liệu nhỏ-vừa.
- **Hiệu suất tương đương:** Nhiều nghiên cứu thực nghiệm cho thấy GRU đạt hiệu suất so sánh được với LSTM trong các bài toán dự báo tài chính [12, 13].

### 2.2.3. Attention pooling

Thay vì chỉ sử dụng trạng thái ẩn cuối cùng (last hidden state) của GRU, đồ án sử dụng cơ chế attention pooling để tổng hợp thông tin từ tất cả các bước thời gian. Cơ chế attention học trọng số cho từng bước thời gian:

$$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T} \exp(e_k)}$$

$$e_t = v^T \tanh(W_a h_t + b_a)$$

$$c = \sum_{t=1}^{T} \alpha_t h_t$$

trong đó $h_t$ là trạng thái ẩn tại bước $t$, $W_a$ và $v$ là các tham số học được, và $c$ là vector ngữ cảnh (context vector) — biểu diễn tổng hợp có trọng số của toàn bộ chuỗi [14, 15].

Cơ chế này cho phép mô hình tập trung vào các giai đoạn cụ thể trong chuỗi đầu vào thay vì coi trọng tất cả các bước thời gian như nhau — đặc biệt quan trọng trong dữ liệu tài chính nơi một số khoảng thời gian chứa nhiều thông tin hơn các khoảng khác.

### 2.2.4. Variational Dropout

Để tăng khả năng tổng quát hóa, đồ án sử dụng variational dropout (hay locked dropout) [16]. Khác với dropout thông thường (sinh mask ngẫu nhiên độc lập cho mỗi phần tử), variational dropout tạo một mask duy nhất cho mỗi mẫu và phát sóng (broadcast) mask đó qua tất cả các bước thời gian:

$$\text{mask} \sim \text{Bernoulli}(1-p), \quad \text{shape: } (B, 1, F)$$

$$x_{\text{dropped}} = x \odot \text{mask} / (1-p)$$

Việc sử dụng cùng một mask trên toàn bộ chuỗi giúp bảo toàn tính tương quan thời gian (temporal correlation) — đặc điểm quan trọng của dữ liệu tài chính mà RNN cần khai thác.

## 2.3. LightGBM

### 2.3.1. Gradient Boosting Decision Tree (GBDT)

Gradient Boosting là phương pháp ensemble xây dựng tuần tự các cây quyết định (decision tree), trong đó mỗi cây mới được huấn luyện để sửa lỗi của tổ hợp cây trước đó [17]. Cho hàm mất mát $L(y, F(x))$, tại vòng lặp $m$:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

trong đó $h_m(x)$ là cây mới học để xấp xỉ gradient âm của hàm mất mát, và $\eta$ là learning rate (shrinkage).

### 2.3.2. Đặc điểm của LightGBM

LightGBM (Light Gradient Boosting Machine), được đề xuất bởi Ke et al. (2017) [18], là một triển khai hiệu quả cao của GBDT với các cải tiến chính:

**Leaf-wise growth strategy:** Khác với XGBoost sử dụng level-wise growth (cân bằng độ sâu), LightGBM phát triển theo leaf-wise — luôn phân chia lá có giảm lỗi lớn nhất. Cách tiếp cận này tạo ra cây không cân bằng nhưng giảm lỗi nhanh hơn đáng kể, đặc biệt trên tập dữ liệu lớn.

**Gradient-based One-Side Sampling (GOSS):** Giữ lại tất cả các mẫu có gradient lớn (đóng góp nhiều vào việc học) và lấy mẫu ngẫu nhiên một tỷ lệ nhỏ các mẫu có gradient nhỏ. Điều này giảm số lượng mẫu cần xử lý mà vẫn giữ được phân phối gradient chính xác.

**Exclusive Feature Bundling (EFB):** Gộp các đặc trưng loại trừ lẫn nhau (exclusive features) thành một đặc trưng duy nhất, giảm số lượng đặc trưng cần xét mà không làm mất thông tin.

**Histogram-based splitting:** Thay vì duyệt tất cả các ngưỡng phân chia khả dĩ, LightGBM rời rạc hóa giá trị đặc trưng thành các bin (histogram) và duyệt các bin — giảm độ phức tạp từ $O(\text{data} \times \text{features})$ xuống $O(\text{bins} \times \text{features})$.

### 2.3.3. Ưu điểm trong dự báo tài chính

LightGBM đặc biệt phù hợp với dự báo tài chính vì [19, 20]:

- **Xử lý tốt đặc trưng dạng bảng (tabular features):** Các chỉ báo kỹ thuật, tỷ lệ, và thống kê trượt là dạng dữ liệu bảng mà GBDT xử lý xuất sắc.
- **Khả năng nắm bắt tương tác đặc trưng (feature interaction):** Tự động học các mối quan hệ phi tuyến giữa các đặc trưng kỹ thuật.
- **Chống overfitting tích hợp:** Hỗ trợ regularization L1/L2, subsampling, feature subsampling, và early stopping.
- **Tốc độ nhanh:** GOSS và EFB giúp huấn luyện nhanh hơn XGBoost 5–20 lần trên cùng dữ liệu.
- **Khả năng diễn giải:** Feature importance và SHAP values có thể được sử dụng để hiểu mô hình.

## 2.4. Hybrid Stacking: Kết hợp GRU và LightGBM

### 2.4.1. Nguyên lý stacking

Stacking (hay stacked generalization) là kỹ thuật ensemble kết hợp nhiều mô hình cơ sở (base learners) thông qua một meta-learner [21]. Trong kiến trúc two-level stacking:

**Level 0 (Base learners):** Các mô hình độc lập học trên dữ liệu gốc.

**Level 1 (Meta-learner):** Học trên đầu ra của base learners (hoặc kết hợp đầu ra base learners với đặc trưng gốc).

### 2.4.2. Kiến trúc hybrid GRU + LightGBM

Trong đồ án này, kiến trúc stacking có dạng đặc biệt:

1. **GRU đóng vai trò feature extractor:** Mạng GRU xử lý chuỗi đầu vào (48 giờ gần nhất của OHLCV chuẩn hóa) và tạo ra vector đặc trưng tuần tự thông qua attention pooling. Vector này phản ánh các mẫu hình thời gian (temporal patterns) mà các đặc trưng kỹ thuật đơn lẻ không thể nắm bắt.

2. **Giảm chiều bằng PCA:** Hidden states của GRU (64 chiều) được giảm xuống 16 thành phần chính (principal components) bằng PCA, loại bỏ nhiễu và giảm chiều không gian đặc trưng.

3. **Ghép nối đặc trưng (feature concatenation):** Các thành phần chính GRU được ghép với các đặc trưng kỹ thuật tĩnh (static features: RSI, ATR, ADX, MACD, v.v.) tạo thành ma trận đặc trưng lai.

4. **LightGBM đóng vai trò phân loại cuối cùng:** LightGBM phân loại trên ma trận đặc trưng lai, tận dụng khả năng xử lý đặc trưng bảng và học tương tác phi tuyến.

Kiến trúc này tương tự với phương pháp được Ju et al. (2020) [9] đề xuất, trong đó các mô hình RNN đóng vai trò base learner và các mô hình tree-based đóng vai trò meta-learner. Nghiên cứu của Liu et al. (2024) [22] cũng áp dụng GRU-LightGBM hybrid cho dự báo chuỗi thời gian với kết quả tích cực.

### 2.4.3. Lợi ích của kiến trúc hybrid

- **Bổ sung thông tin:** GRU nắm bắt mẫu hình tuần tự (sequential patterns) trong chuỗi giá, trong khi LightGBM xử lý tốt các đặc trưng kỹ thuật dạng bảng.
- **Giảm variance:** Ensemble hai họ mô hình khác nhau (neural net + tree) thường giảm variance so với mô hình đơn lẻ.
- **Hiệu quả tính toán:** GRU chỉ cần xử lý một lần để trích xuất hidden states, sau đó LightGBM (nhanh hơn nhiều) thực hiện phân loại.

## 2.5. Walk-Forward Validation

### 2.5.1. Vấn đề với cross-validation truyền thống

K-Fold cross-validation truyền thống chia ngẫu nhiên dữ liệu thành K phần — phương pháp này **không phù hợp** với dữ liệu chuỗi thời gian vì [13, 23]:

- **Rò rỉ thông tin (information leakage):** Dữ liệu tương lai có thể nằm trong tập huấn luyện trong khi dữ liệu quá khứ nằm trong tập kiểm tra.
- **Tự tương quan chuỗi:** Các mẫu gần nhau trong thời gian có tương quan cao, làm cho K-Fold đánh giá quá lạc quan.
- **Label lookahead:** Nhãn tại thời điểm $t$ có thể phụ thuộc vào dữ liệu từ $t$ đến $t + h$ (horizon), nên nếu mẫu tại $t+h$ nằm trong tập huấn luyện sẽ gây rò rỉ.

### 2.5.2. Sliding walk-forward validation

Walk-forward validation (hay rolling-origin evaluation) là phương pháp chuẩn cho dữ liệu chuỗi thời gian [24]:

1. **Cửa sổ huấn luyện (training window):** Sử dụng $N_{\text{train}}$ thanh giá (bars) liên tiếp để huấn luyện.
2. **Cửa sổ kiểm tra (test window):** Đánh giá mô hình trên $N_{\text{test}}$ thanh giá tiếp theo.
3. **Trượt (slide):** Tiến cửa sổ thêm $N_{\text{step}}$ thanh giá và lặp lại.

### 2.5.3. Purge và Embargo

Để ngăn chặn rò rỉ thông tin tại biên giữa tập huấn luyện và tập kiểm tra, López de Prado (2018) [13] đề xuất hai cơ chế:

**Purge:** Loại bỏ $p$ thanh giá cuối cùng của tập huấn luyện. Vì nhãn tại thời điểm $t$ sử dụng dữ liệu từ $t$ đến $t + h$ (horizon), mẫu gần biên giới train/test có nhãn phụ thuộc vào dữ liệu test.

$$\text{adjusted\_train\_end} = \text{raw\_train\_end} - p$$

**Embargo:** Bỏ qua $e$ thanh giá đầu tiên của tập kiểm tra, tạo khoảng đệm bổ sung:

$$\text{adjusted\_test\_start} = \text{raw\_test\_start} + p + e$$

Tổng khoảng cách giữa tập huấn luyện và tập kiểm tra là $2p + e$ thanh giá — đảm bảo không có chồng chéo thông tin giữa hai tập.

### 2.5.4. Event-time purge

Trong đồ án, purge được nâng cấp thành **event-time purge**: thay vì loại bỏ cố định $p$ thanh giá, chỉ giữ lại các mẫu huấn luyện có nhãn kết thúc sự kiện (event_end) trước biên kiểm tra. Phương pháp này chính xác hơn vì Triple Barrier labels có thời gian sự kiện biến đổi [13].

## 2.6. Phương pháp Triple Barrier Labeling

### 2.6.1. Nguyên lý

Triple Barrier Method, được đề xuất bởi López de Prado (2018) [13], gán nhãn cho mỗi mẫu dựa trên chạm barrier đầu tiên trong ba barrier:

1. **Upper barrier (take-profit):** $U_i = \text{close}_i + \text{tp\_mult} \times \text{ATR}_i$
2. **Lower barrier (stop-loss):** $L_i = \text{close}_i - \text{sl\_mult} \times \text{ATR}_i$
3. **Vertical barrier (time horizon):** Sau $h$ thanh giá

Nhãn được xác định:

- **Long (+1):** Giá chạm upper barrier trước (lợi nhuận tiềm năng đạt mức kỳ vọng).
- **Short (−1):** Giá chạm lower barrier trước (rủi ro vượt ngưỡng).
- **Hold (0):** Không chạm barrier nào trong horizon, hoặc chạm cả hai cùng thanh giá (ambiguous).

### 2.6.2. ATR-based dynamic barriers

Đồ án sử dụng Average True Range (ATR) làm đơn vị đo khoảng cách barrier thay vì giá trị cố định (fixed ticks). Điều này có hai lợi ích:

- **Thích ứng với biến động:** Barrier tự động mở rộng khi thị trường biến động mạnh và thu hẹp khi thị trường bình ổn.
- **Tương đương giữa các chế độ thị trường:** P&L tiềm năng (tính bằng ATR) có ý nghĩa kinh tế tương đương bất kể mức giá tuyệt đối.

### 2.6.3. Sample weighting: Average Uniqueness

Vì các nhãn Triple Barrier có thời gian sự kiện chồng chéo (overlapping), cần xử lý vấn đề mẫu quá dày đặc (high concurrency). López de Prado đề xuất **average uniqueness** weight [13]:

$$w_i = \frac{1}{|T_i|} \sum_{t \in T_i} \frac{1}{c(t)}$$

trong đó $T_i$ là tập các thanh giá mà mẫu $i$ đang hoạt động, và $c(t)$ là số lượng mẫu đang hoạt động tại thời điểm $t$. Các mẫu có nhiều chồng chéo nhận trọng số thấp hơn, giảm bias do duplicate information.

### 2.6.4. Censored labels

Các mẫu trong khoảng $h$ thanh giá cuối cùng của chuỗi dữ liệu không có đủ dữ liệu tương lai để đánh giá barrier → được đánh dấu là censored (−2) và loại bỏ khỏi tập huấn luyện. Điều này ngăn chặn nhãn Hold giả tạo do thiếu dữ liệu.

## 2.7. Tổng kết chương

Chương này đã trình bày nền tảng lý thuyết cho đồ án: đặc thù của chuỗi thời gian tài chính, kiến trúc GRU với attention pooling và variational dropout, LightGBM và các cải tiến, nguyên lý stacking hybrid, walk-forward validation với purge/embargo, và Triple Barrier labeling. Chương tiếp theo sẽ mô tả cách thức thu thập và tiền xử lý dữ liệu thực tế.

## Tài liệu tham khảo chương này

[1] Ke, Z., et al. (2025). "A comprehensive survey of deep learning for time series forecasting." *Artificial Intelligence Review*, 58, Article 175.

[2] Harvey, C.R. (2017). "Presidential Address: The Scientific Outlook in Financial Economics." *Journal of Finance*, 72(4), 1399–1440.

[3] López de Prado, M. (2018). "The 10 Reasons Most Machine Learning Funds Fail." *Journal of Portfolio Management*, 44(6), 120–133.

[4] Cont, R. (2001). "Empirical properties of asset returns: stylized facts and statistical issues." *Quantitative Finance*, 1(2), 223–236.

[5] Fama, E.F. (1970). "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383–417.

[6] Krauss, C., Do, X.A., & Huck, N. (2017). "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689–702.

[7] Taylor, S.J. (2007). *Modelling Financial Time Series* (2nd ed.). World Scientific.

[8] Patel, J., et al. (2015). "Predicting stock market index using fusion of machine learning techniques." *Expert Systems with Applications*, 42(4), 2162–2172.

[9] Ju, Y., et al. (2020). "An improved Stacking framework for stock index prediction." *Physica A*, 541, 122272.

[10] Chung, J., et al. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." *arXiv:1412.3555*.

[11] Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735–1780.

[12] Fischer, T. & Krauss, C. (2018). "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654–669.

[13] López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

[14] Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473*.

[15] Qin, Y., et al. (2017). "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction." *IJCAI*.

[16] Gal, Y. & Ghahramani, Z. (2016). "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks." *Advances in Neural Information Processing Systems 29*.

[17] Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189–1232.

[18] Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NIPS 30*, pp. 3149–3157.

[19] Sun, X., et al. (2020). "A novel cryptocurrency price trend forecasting model based on LightGBM." *Finance Research Letters*, 32, 101084.

[20] Wang, Y., et al. (2024). "Enhancing financial time series forecasting: A hybrid approach with LightGBM." *Engineering Applications of Artificial Intelligence*, 133, 108510.

[21] Wolpert, D.H. (1992). "Stacked Generalization." *Neural Networks*, 5(2), 241–259.

[22] Liu, Y., et al. (2024). "GRU-LightGBM for Short-Shelf-Life Products Time Series Prediction." *Applied Sciences*, 14(2), 866.

[23] Bailey, D.H., et al. (2014). "Pseudo-Mathematics and Financial Charlatanism." *Notices of the AMS*, 61(5), 458–471.

[24] Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.
