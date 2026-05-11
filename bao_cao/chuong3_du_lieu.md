# Chương 3. Dữ liệu và tiền xử lý

## 3.1. Nguồn dữ liệu

Dữ liệu sử dụng trong đồ án là dữ liệu XAU/USD được xử lý thành thanh OHLCV khung 1 giờ. XAU/USD biểu diễn giá vàng theo đô la Mỹ. Vàng là thị trường có thanh khoản cao và chịu tác động của nhiều yếu tố như lãi suất thực, lạm phát kỳ vọng, rủi ro địa chính trị, nhu cầu đầu tư và hoạt động mua bán của ngân hàng trung ương [3].

Bảng mô tả bộ dữ liệu:

| Thuộc tính | Giá trị |
|---|---|
| Tài sản | XAU/USD |
| Khung thời gian | H1 |
| Giai đoạn dữ liệu | 2018-01-01 đến 2026-04-30 |
| Dạng xử lý | OHLCV |
| Múi giờ | America/New_York |
| Bài toán | Phân loại Short/Hold/Long |

Dữ liệu tài chính theo chuỗi thời gian không được xáo trộn ngẫu nhiên khi đánh giá. Mọi bước xử lý cần giữ thứ tự thời gian và tránh sử dụng thông tin tương lai.

## 3.2. Từ tick đến OHLCV

Nếu dữ liệu gốc ở dạng tick, pipeline tổng hợp thành OHLCV theo từng giờ:

- Open: giá đầu tiên trong giờ.
- High: giá cao nhất trong giờ.
- Low: giá thấp nhất trong giờ.
- Close: giá cuối cùng trong giờ.
- Volume: tổng khối lượng hoặc proxy volume trong giờ.

Trường hợp có bid/ask, giá mid có thể được tính bằng:

```text
mid_price = (bid + ask) / 2
```

Việc dùng OHLCV H1 giúp cân bằng giữa chi tiết intraday và nhiễu microstructure. Khung nhỏ hơn như M1/M5 thường nhiễu hơn và nhạy với spread/slippage; khung lớn hơn như D1 có ít mẫu hơn.

## 3.3. Kiểm tra chất lượng dữ liệu

Pipeline kiểm tra các điều kiện sau:

1. Timestamp tăng dần.
2. Không có timestamp trùng lặp.
3. Các cột OHLCV bắt buộc tồn tại.
4. Không có giá âm hoặc bất hợp lý.
5. Ghi nhận gap thời gian do cuối tuần, ngày nghỉ hoặc thiếu dữ liệu.
6. Kiểm tra thống kê cơ bản của close, return, volume và ATR.

Các gap cuối tuần không được nội suy thành thanh giả vì thị trường thực tế không giao dịch. Việc nội suy có thể tạo ra dữ liệu nhân tạo và làm sai feature kỹ thuật.

## 3.4. Feature engineering causal

Feature engineering là bước chuyển OHLCV thành ma trận đặc trưng cho mô hình. Nguyên tắc của đề tài là chỉ tạo feature từ quá khứ và hiện tại. Không đưa vào mô hình các cột có thông tin tương lai như label, touched_bar, event_end, upper_barrier, lower_barrier hoặc sample_weight.

Sau lần chạy mới nhất, feature whitelist còn 21 đặc trưng model-facing. Các feature bị loại do importance thấp gồm:

```text
regime_strength
upper_wick_ratio
lower_wick_ratio
volume_zscore_20
```

Việc giảm feature nhằm giảm nhiễu và giúp báo cáo dễ bảo vệ hơn. Tuy nhiên, các indicator gốc không nhất thiết bị xóa khỏi code; chỉ whitelist model-facing được tinh chỉnh.

## 3.5. Nhóm đặc trưng sử dụng

### 3.5.1. Trend

Các đặc trưng xu hướng mô tả quan hệ giữa giá và đường trung bình:

- `ema34_vs_ema89`: khoảng cách EMA34 so với EMA89.
- `close_vs_ema_34`: vị trí giá đóng cửa so với EMA34.
- `ema_slope_20`: độ dốc EMA20.
- `adx_14`: cường độ xu hướng.

ADX và các mẫu kỹ thuật có thể được nghiên cứu dưới góc nhìn thống kê, như hướng tiếp cận của Lo, Mamaysky và Wang [11]. Trong pipeline, chúng không được dùng như quy tắc vào lệnh cứng mà là feature cho mô hình.

### 3.5.2. Momentum

Momentum phản ánh tốc độ thay đổi giá:

- `return_1h`, `return_4h`: log-return ngắn hạn.
- `rsi_14`: dao động quá mua/quá bán.
- `macd_hist_atr`: MACD histogram chuẩn hóa theo ATR.

Log-return được dùng vì có tính cộng dồn theo thời gian và thường phù hợp hơn return tỷ lệ đơn giản khi phân tích chuỗi giá.

### 3.5.3. Volatility

Biến động là thành phần quan trọng trong cả feature và labeling:

- `atr_pct_close`: ATR chia cho close.
- `atr_ratio`: ATR hiện tại so với mức nền.
- `high_low_range_20`: range cao-thấp rolling.
- `price_dist_ratio`: khoảng cách giá được chuẩn hóa.

ATR giúp barrier thích nghi với biến động: khi thị trường biến động mạnh, TP/SL rộng hơn; khi biến động thấp, TP/SL hẹp hơn nhưng vẫn có floor để tránh barrier quá nhỏ.

### 3.5.4. Price position

Nhóm này mô tả vị trí giá trong cấu trúc gần đây:

- `price_position_20`: vị trí close trong range rolling 20.
- `pivot_position`: vị trí so với pivot.
- `vwap`: giá trung bình có trọng số volume.

Các feature này giúp mô hình nhận biết giá đang ở gần vùng cao/thấp tương đối hay gần trung tâm range.

### 3.5.5. Session

XAU/USD có hành vi khác nhau theo phiên giao dịch. Pipeline dùng các biến session:

- `sess_asia`
- `sess_london`
- `sess_ny_am`
- `sess_ny_pm`

Session features giúp mô hình phân biệt các giai đoạn thanh khoản và biến động trong ngày.

## 3.6. Warm-up và xử lý missing values

Các chỉ báo kỹ thuật cần một số lượng quan sát tối thiểu. Ví dụ EMA26 hoặc rolling window 20 chưa có ý nghĩa ở các dòng đầu. Pipeline loại bỏ warm-up rows có feature model-facing chưa hoàn chỉnh. Sau khi tạo feature, hệ thống kiểm tra null/NaN để đảm bảo dữ liệu đưa vào training sạch.

## 3.7. Gán nhãn dữ liệu

Stage 3 đọc feature matrix và OHLCV để tạo nhãn triple-barrier. Cấu hình đang dùng:

```toml
[labels]
atr_tp_multiplier = 2.0
atr_sl_multiplier = 2.0
horizon_bars = 24
```

Với khung H1, horizon 24 tương ứng tối đa 24 giờ. TP/SL đối xứng 2.0 ATR giúp tránh thiên lệch ban đầu về Long hoặc Short. Các mẫu bị censored không phù hợp được loại khỏi training.

Phân phối nhãn trong phiên gần nhất:

```text
Short: 43.6%
Hold :  9.0%
Long : 47.4%
```

Hold thấp cho thấy đa số sự kiện chạm một trong hai barrier trong 24 giờ. Đã thử horizon 48 nhưng Hold giảm còn khoảng 1.5%, nên horizon 24 được giữ lại vì ít làm bài toán lệch khỏi thiết kế 3 lớp hơn.

## 3.8. Tập dữ liệu huấn luyện cuối cùng

Bộ dữ liệu huấn luyện cuối cùng gồm:

- Timestamp để giữ thứ tự thời gian và join dữ liệu.
- 21 feature model-facing.
- Label Short/Hold/Long.
- Event metadata như event_end và sample_weight phục vụ purge/embargo và weighting.

Các metadata này không được đưa vào X khi huấn luyện. Chúng chỉ phục vụ validation và đánh giá.

## 3.9. Tổng kết chương

Chương này trình bày quy trình từ dữ liệu XAU/USD H1 đến ma trận feature-label dùng cho mô hình. Các quyết định quan trọng gồm: giữ thứ tự thời gian, không nội suy cuối tuần, tạo feature causal, loại warm-up rows, gán nhãn bằng ATR triple-barrier và kiểm tra phân phối lớp. Đây là nền tảng để Stage 4 huấn luyện mô hình mà không bị rò rỉ thông tin tương lai.

## 3.10. Data contract giữa các stage

Pipeline hoạt động ổn định nhờ mỗi stage có data contract rõ ràng. Nếu một stage thay đổi schema mà stage sau không biết, kết quả có thể sai hoặc pipeline dừng.

| Stage | Input chính | Output chính | Điều kiện bắt buộc |
|---|---|---|---|
| Stage 1 | Raw/tick/OHLCV source | `ohlcv.parquet` | timestamp sorted, OHLCV hợp lệ |
| Stage 2 | OHLCV | `features.parquet`, feature list | ATR tồn tại, feature causal, không null sau warm-up |
| Stage 3 | Features + OHLCV | `labels.parquet` | có `atr_14`, label ∈ {-1,0,1,-2}, có `event_end` |
| Stage 4 | Labels + features | predictions, model, metrics | không dùng metadata làm feature |
| Stage 5 | Predictions + OHLCV | backtest results | TP/SL backtest khớp label |
| Stage 6 | Metrics + backtest | report | đọc đúng artifacts mới nhất |

Data contract giúp báo cáo giải thích được vì sao thay đổi feature phải rerun Stage 2 trở đi, còn thay đổi label phải rerun Stage 3 trở đi.

## 3.11. Cột bị loại khỏi feature model-facing

Một số cột bị loại khỏi X training dù vẫn xuất hiện trong dataset:

| Nhóm cột | Ví dụ | Lý do loại |
|---|---|---|
| Định danh/thời gian | `timestamp` | dùng để sort/join, không phải tín hiệu trực tiếp |
| Target | `label` | chính là nhãn cần dự báo |
| Barrier metadata | `upper_barrier`, `lower_barrier`, `touched_bar`, `event_end` | được tạo bằng thông tin tương lai |
| Raw OHLCV | `open`, `high`, `low`, `close`, `volume` | tránh raw price scale và leakage không kiểm soát |
| Helper | `atr_14` | dùng tạo barrier; dùng feature normalized thay thế |

Việc loại các cột này là một lớp bảo vệ leakage. Nếu đưa barrier hoặc event_end vào mô hình, mô hình có thể học trực tiếp thông tin của quá trình gán nhãn, khiến kết quả không còn ý nghĩa.

## 3.12. Danh sách feature model-facing hiện tại

Feature whitelist sau pruning gồm các nhóm sau.

### Trend và trend quality

```text
ema34_vs_ema89
close_vs_ema_34
adx_14
ema_slope_20
```

### Momentum

```text
return_1h
return_4h
macd_hist_atr
rsi_14
```

### Volatility và range

```text
atr_pct_close
atr_ratio
high_low_range_20
price_dist_ratio
```

### Price position

```text
price_position_20
pivot_position
vwap
```

### Candle/session và feature phụ trợ còn giữ

```text
candle_body_ratio
sess_asia
sess_london
sess_ny_am
sess_ny_pm
```

Tập feature này không nhằm bao phủ mọi chỉ báo kỹ thuật có thể có. Nó được chọn để cân bằng giữa khả năng biểu diễn, tính giải thích và rủi ro overfit.

## 3.13. Kiểm tra phân phối feature

Trước khi train, cần kiểm tra:

1. Feature có NaN hoặc infinite không.
2. Feature có variance gần 0 không.
3. Feature có outlier cực đoan không.
4. Feature có tương quan quá cao với feature khác không.
5. Feature có bị tính bằng thông tin tương lai không.

Đặc biệt trong tài chính, outlier không luôn là lỗi. Spike giá có thể là sự kiện thật. Vì vậy pipeline không nên xóa outlier một cách tùy tiện; thay vào đó cần ghi nhận và dùng mô hình/regularization phù hợp.

## 3.14. Phân phối nhãn và ý nghĩa

Phân phối nhãn mới nhất:

| Lớp | Tỷ lệ | Diễn giải |
|---|---:|---|
| Short | 43.6% | Giá thường chạm lower barrier trước trong các sự kiện này |
| Hold | 9.0% | Ít sự kiện không chạm barrier trong 24 giờ |
| Long | 47.4% | Giá thường chạm upper barrier trước trong các sự kiện này |

Tỷ lệ Hold thấp là điểm cần lưu ý. Nó có thể khiến bài toán gần với binary Long/Short hơn mong muốn. Tuy nhiên, giữ Hold vẫn có ý nghĩa vì lớp này đại diện cho tình huống không nên giao dịch hoặc tín hiệu không rõ ràng. Thay vì bỏ Hold ngay, báo cáo phân tích điểm yếu của lớp này qua F1 và confusion matrix.

## 3.15. Vì sao không đổi sang binary ngay

Chuyển sang binary Long/Short có thể làm metric đẹp hơn, nhưng sẽ thay đổi bản chất bài toán. Nếu bỏ Hold, hệ thống bị ép đưa ra tín hiệu giao dịch ở mọi thời điểm. Trong thực tế, không giao dịch cũng là một quyết định quan trọng.

Ngoài ra, code hiện tại được thiết kế cho multiclass:

- Class order `[-1, 0, 1]`.
- Report per-class Short/Hold/Long.
- Confusion matrix ba lớp.
- Stacking probability features ba cột mỗi base learner.
- Backtest logic nhận tín hiệu Hold như trạng thái không vào lệnh.

Vì vậy, binary là refactor lớn và không phù hợp nếu mục tiêu hiện tại là hoàn thiện báo cáo ổn định.

## 3.16. Kết luận chi tiết về dữ liệu

Chất lượng dữ liệu trong đề tài không chỉ là “không thiếu dòng”. Nó bao gồm tính đúng của thời gian, tính causal của feature, tính hợp lý của nhãn và sự nhất quán giữa stage. Nếu dữ liệu bị leakage, mô hình có thể đạt kết quả cao nhưng không có giá trị. Do đó, chương dữ liệu cần được xem là nền tảng của toàn bộ luận văn, không chỉ là phần mô tả nguồn dữ liệu.
