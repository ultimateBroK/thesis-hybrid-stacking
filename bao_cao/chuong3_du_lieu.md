# Chương 3. Dữ liệu và tiền xử lý

## 3.1. Nguồn dữ liệu

### 3.1.1. Mô tả dữ liệu gốc

Dữ liệu được sử dụng trong đồ án là dữ liệu tick thô của cặp tiền XAU/USD (vàng/đô la Mỹ), thu thập từ nguồn dữ liệu thị trường ngoại hối. Dữ liệu tick ghi lại mỗi giao dịch thực tế bao gồm:

- **Timestamp:** Thời điểm giao dịch chính xác đến mili-giây.
- **Bid/Ask:** Giá mua/bán tại thời điểm giao dịch.
- **Volume:** Khối lượng giao dịch (nếu có).

Dữ liệu OHLCV (Open-High-Low-Close-Volume) sau khi xử lý sử dụng các thông số:

| Thông số | Giá trị |
|---|---|
| Cặp tiền | XAU/USD |
| Khung thời gian | 1 giờ (1H) |
| Múi giờ thị trường | America/New_York |
| Ngày bắt đầu | 2018-01-01 |
| Ngày kết thúc | 2026-04-30 |
| Tick size | 0.01 USD |
| Contract size | 100 oz/lot |

Khoảng thời gian 8+ năm bao gồm nhiều chế độ thị trường: xu hướng tăng mạnh (2020), dao động ngang (2021–2022), và xu hướng mới (2023–2025), đảm bảo mô hình được thử thách trên đủ loại điều kiện thị trường [1].

### 3.1.2. Lý do chọn khung thời gian 1H

Khung thời gian 1 giờ được chọn vì:

1. **Cân bằng giữa độ chi tiết và nhiễu:** Khung 1H cung cấp đủ chi tiết để nắm bắt biến động trong ngày mà không bị nhiễu ngắn hạn như các khung thời gian nhỏ hơn (M15, M5).
2. **Phù hợp với mục tiêu dự báo ngắn hạn:** Horizon 24 giờ (24 thanh giá 1H) tương ứng với khoảng thời gian giữ vị thế 1 ngày giao dịch.
3. **Số lượng mẫu đủ lớn:** Khoảng 72,000 thanh giá trong 8+ năm, đủ cho huấn luyện deep learning.
4. **Thực tế giao dịch:** Nhiều nhà giao dịch swing và day-trader sử dụng khung 1H làm khung chính [2].

## 3.2. Từ Tick đến OHLCV

### 3.2.1. Quy trình tổng hợp

Dữ liệu tick thô được tổng hợp thành thanh giá OHLCV theo các bước:

1. **Gom nhóm theo khung thời gian:** Tick được nhóm theo giờ (từ 00:00 đến 23:59, múi giờ New York).
2. **Tính toán OHLCV:**
   - **Open:** Giá mid (trung bình bid/ask) của tick đầu tiên trong giờ.
   - **High:** Giá mid cao nhất trong giờ.
   - **Low:** Giá mid thấp nhất trong giờ.
   - **Close:** Giá mid của tick cuối cùng trong giờ.
   - **Volume:** Tổng khối lượng giao dịch trong giờ.

3. **Xử lý giờ trống:** Các giờ không có giao dịch (cuối tuần, một số giờ giao dịch chậm) không tạo thanh giá — duy trì tính liên tục thực tế.

### 3.2.2. Xử lý múi giờ

Thị trường forex hoạt động 24/5, nhưng dữ liệu cần được neo vào một múi giờ nhất định để đảm bảo nhất quán. Đồ án sử dụng múi giờ New York (America/New_York) vì:

- New York là trung tâm giao dịch vàng lớn nhất thế giới.
- Phiên New York (13:30–20:00 GMT) có thanh khoản cao nhất cho XAU/USD.
- Việc cố định múi giờ tránh lỗi do thay đổi giờ mùa hè (DST).

## 3.3. Làm sạch dữ liệu

### 3.3.1. Kiểm tra tính toàn vẹn

Pipeline kiểm tra các tiêu chí sau trước khi xử lý:

**Sắp xếp thời gian:** Đảm bảo timestamp tăng dần nghiêm ngặt (strictly ascending). Bất kỳ sự đảo ngược nào đều gây lỗi.

**Duy nhất timestamp:** Mỗi timestamp chỉ xuất hiện một lần. Trùng lặp timestamp gây lỗi khi join feature với label.

**Khoảng trống (gap):** Kiểm tra khoảng cách giữa các thanh giá liên tiếp. Khoảng trống lớn hơn expected (1 giờ) được ghi nhận:

```
Feature input gap check: expected_delta=3600000 ms, gap_count=X, largest_gap=Y bars
```

### 3.3.2. Xử lý dữ liệu thiếu

- **Thanh giá cuối tuần:** Không tạo thanh giả (no interpolation) — thị trường thực tế đóng cửa.
- **Outlier giá:** Kiểm tra variation quá lớn giữa Open/Close (>10% trong 1 giờ) nhưng không loại bỏ — đây có thể là sự kiện thị trường hợp lệ.
- **Volume bằng 0:** Giữ nguyên — thanh khoản thấp là thông tin thị trường thực.

### 3.3.3. Kiểm tra phân phối

Dữ liệu OHLCV được kiểm tra các thống kê mô tả cơ bản:

- Min/Max/Median của giá Close và Volume.
- Tỷ lệ thanh giá có volume bằng 0.
- Phân phối log-return (tỷ suất lợi nhuận log) để kiểm tra normality và fat tails.

## 3.4. Feature Engineering

### 3.4.1. Tổng quan

Feature engineering là bước tạo các đặc trưng (features) từ dữ liệu OHLCV thô để mô hình có thể học. Đồ án áp dụng nguyên tắc: ưu tiên cấu trúc giá và khoảng cách xu hướng hơn việc chồng chất nhiều chỉ báo, tránh biến đổi dư thừa của cùng một tín hiệu [3].

Tổng cộng, pipeline tạo ra các nhóm đặc trưng sau:

### 3.4.2. Nhóm 1: Giá và biến động (Price & Volatility)

**Average True Range (ATR):**

$$\text{ATR}_t = \text{EMA}_{14}\left(\max(H_t - L_t,\; |H_t - C_{t-1}|,\; |L_t - C_{t-1}|)\right)$$

ATR là nền tảng cho barrier labeling và risk management. Chu kỳ 14 giờ (mặc định) cung cấp ước lượng biến động ổn định.

**Log-return:**

$$r_t = \ln\left(\frac{C_t}{C_{t-1}}\right)$$

Log-return tại các chu kỳ 1, 2, 4, 8, 24 giờ (return_1h, return_2h, ..., return_24h) cho phép mô hình nắm bắt xu hướng ở nhiều thời lượng.

**High-Low Range (phạm vi cao-thấp):**

$$\text{hl\_range}_t = \frac{H_t - L_t}{C_t}$$

Phạm vi dao động trong mỗi thanh giá, chuẩn hóa theo giá đóng cửa.

### 3.4.3. Nhóm 2: Xu hướng và chất lượng xu hướng (Trend & Trend Quality)

**ADX (Average Directional Index):** Đo lường sức mạnh xu hướng (không phân biệt hướng). ADX > 25 thường cho thấy xu hướng rõ ràng, ADX < 20 cho thị trường dao động ngang [4].

**EMA Slope:** Độ dốc của EMA(20), tính bằng log-return của EMA:

$$\text{ema\_slope}_t = \ln\left(\frac{\text{EMA}_{20,t}}{\text{EMA}_{20,t-1}}\right)$$

EMA slope dương → xu hướng tăng, âm → xu hướng giảm. Độ lớn slope phản ánh tốc độ xu hướng.

**EMA Crossover:** Tín hiệu giao nhau giữa EMA nhanh và EMA chậm:

$$\text{ema\_cross}_t = \frac{\text{EMA}_{\text{fast},t} - \text{EMA}_{\text{slow},t}}{\text{ATR}_t}$$

Chuẩn hóa bằng ATR để tín hiệu giao nhau có ý nghĩa tương đương giữa các chế độ biến động.

### 3.4.4. Nhóm 3: Dao động (Oscillators)

**RSI (Relative Strength Index):** Dao động đo tốc độ và sự thay đổi của biến động giá:

$$\text{RSI}_t = 100 - \frac{100}{1 + \frac{\text{EMA}_{14}(\text{gain})}{\text{EMA}_{14}(\text{loss})}}$$

RSI > 70 thường cho thấy quá mua (overbought), RSI < 30 cho thấy quá bán (oversold).

**MACD (Moving Average Convergence Divergence):** Khác biệt giữa EMA(12) và EMA(26), với signal line EMA(9):

$$\text{MACD}_t = \text{EMA}_{12}(C_t) - \text{EMA}_{26}(C_t)$$

$$\text{MACD\_signal}_t = \text{EMA}_9(\text{MACD}_t)$$

$$\text{MACD\_hist}_t = \text{MACD}_t - \text{MACD\_signal}_t$$

MACD histogram đặc biệt hữu ích để phát hiện sự suy yếu xu hướng.

### 3.4.5. Nhóm 4: Chế độ thị trường (Market Regime)

**Regime composite:** Đặc trưng tổng hợp phân loại chế độ thị trường dựa trên kết hợp ADX, ATR, và xu hướng giá:

- Regime 0: Dao động ngang (low ADX, low ATR)
- Regime 1: Xu hướng tăng (high ADX, positive slope)
- Regime 2: Xu hướng giảm (high ADX, negative slope)
- Regime 3: Biến động cao (high ATR bất kể hướng)

Biến regime giúp mô hình điều chỉnh chiến lược dự báo theo điều kiện thị trường hiện tại [5].

**Volume Z-score:** Khối lượng giao dịch chuẩn hóa:

$$\text{vol\_z}_t = \frac{V_t - \mu_V}{\sigma_V}$$

Volume z-score cao → thanh khoản bất thường, thường liên quan đến sự kiện thị trường.

### 3.4.6. Nhóm 5: OHLCV chuẩn hóa cho GRU

Để GRU xử lý chuỗi đầu vào ổn định, các giá trị OHLCV gốc được chuẩn hóa:

- **Price normalization:** Close, Open, High, Low được chuẩn hóa theo close cuối chuỗi (relative price).
- **Volume normalization:** Volume được log-transform và chuẩn hóa z-score.

Các đặc trưng này là đầu vào trực tiếp cho GRU (không phải cho LightGBM), giúp mạng nơ-ron học mẫu hình thời gian từ chuỗi giá thô.

### 3.4.7. Loại bỏ warm-up rows

Các hàng đầu tiên có chỉ báo kỹ thuật chưa đủ dữ liệu lịch sử (ví dụ: EMA(26) cần ít nhất 26 thanh giá) → giá trị NaN hoặc không chính xác. Pipeline loại bỏ các hàng này:

```
Dropped X warm-up rows with incomplete model-facing features
```

### 3.4.8. Lọc đặc trưng tương quan cao

Các đặc trưng có tương quan tuyệt đối > 0.75 với đặc trưng khác bị loại bỏ để tránh đa cộng tuyến (multicollinearity), giữ lại đặc trưng giải thích cao nhất [3]. Ngưỡng này được cấu hình qua `correlation_threshold` trong `config.toml`.

## 3.5. Kiểm tra chất lượng dữ liệu (Data Quality Validation)

### 3.5.1. Kiểm tra ở mỗi giai đoạn

Pipeline thực hiện validation tại mỗi biên giai đoạn (stage boundary):

| Giai đoạn | Kiểm tra |
|---|---|
| OHLCV → Features | Timestamp unique, sorted, required columns, gap analysis |
| Features → Labels | Timestamp unique, ATR column exists, ATR statistics |
| Labels → Training | Censored labels dropped, NaN regression targets removed |

### 3.5.2. Kiểm tra phân phối ATR

ATR là thành phần cốt lõi của Triple Barrier labeling. Pipeline ghi nhận thống kê ATR:

```
ATR stats (atr_14): min=X, median=Y, p5=Z, p95=W, below_min_atr=V%
```

Nếu ATR quá thấp (dưới `min_atr`), barrier sẽ quá hẹp → nhãn Hold nhiều → không hữu ích cho giao dịch. Floor `min_atr` đảm bảo barrier tối thiểu.

### 3.5.3. Kiểm tra phân phối nhãn

Sau khi tạo nhãn, pipeline ghi nhận tỷ lệ mỗi lớp:

```
Class -1 (Short): X% 
Class  0 (Hold):  Y%
Class +1 (Long): Z%
```

Phân phối lớp mất cân bằng nghiêm trọng (ví dụ: Hold > 80%) cảnh báo rằng barrier không phù hợp với chế độ thị trường hiện tại [6].

### 3.5.4. Kiểm tra tính khả thi kinh tế của nhãn (Label Profitability)

Pipeline kiểm tra tỷ lệ nhãn Long/Short thực sự có lãi sau chi phí giao dịch (spread + slippage + commission). Nếu cả hai lớp đều dưới 60%, cảnh báo được phát ra:

```
LABEL PROFITABILITY LOW: Long X%, Short Y% -- labels may not be economically useful
```

Kiểm tra này đảm bảo nhãn có ý nghĩa kinh tế, không chỉ ý nghĩa thống kê.

## 3.6. Tổng kết chương

Chương này đã trình bày toàn bộ quy trình từ dữ liệu tick thô đến feature matrix sẵn sàng cho huấn luyện: tổng hợp OHLCV, làm sạch, tạo 5 nhóm đặc trưng (price/volatility, trend, oscillators, regime, normalized OHLCV), và validation đa lớp. Các quyết định thiết kế (ATR-based barriers, correlation filtering, warm-up removal) đều nhằm đảm bảo chất lượng dữ liệu đầu vào cho mô hình.

## Tài liệu tham khảo chương này

[1] Cordero, C.A., et al. (2017). "Predicting the XAU-USD Foreign Exchange Prices using Machine Learning." *Master's thesis*.

[2] Amini, A. & Kalantari, R. (2024). "Gold price prediction by a CNN-Bi-LSTM model." *PLOS ONE*, 19(3), e0298426.

[3] Oikonomou, K. & Damigos, D. (2025). "Assets Forecasting with Feature Engineering and Transformation Techniques using LightGBM." *arXiv:2501.07580*.

[4] Lo, A.W., Mamaysky, H., & Wang, J. (2000). "Foundations of Technical Analysis." *The Journal of Finance*, 55(4), 1705–1765.

[5] Arian, H., et al. (2025). "Regime-Aware LightGBM for Stock Market Forecasting." *Electronics*, 15(6), 1334.

[6] Kim, H., et al. (2025). "Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data." *arXiv:2504.02249*.
