# Chương 6. Minh họa ứng dụng

## 6.1. Mục đích của chương

Chương này minh họa cách mô hình hybrid có thể được áp dụng trong bối cảnh giao dịch thực tế thông qua một backtest demo. **Cần nhấn mạnh:** kết quả backtest trong chương này **không** là bằng chứng chính cho hiệu quả của phương pháp đề xuất. Mục tiêu chính của đồ án là đánh giá hiệu suất dự báo (Chương 5), không phải xây dựng hệ thống giao dịch [1].

## 6.2. Thiết lập Backtest

### 6.2.1. Thông số giao dịch

| Thông số | Giá trị | Giải thích |
|---|---|---|
| Vốn ban đầu | $10,000 | Tài khoản demo |
| Đòn bẩy (leverage) | 10:1 | Đòn bẩy phổ biến cho vàng |
| Spread | 35 ticks (0.35 USD) | Spread trung bình XAU/USD |
| Slippage | 5 ticks (0.05 USD) | Trượt giá ước tính |
| Commission | $10/lot/round-trip | Phí broker |
| Lot size | 0.01 lot | Vị thế tối thiểu |
| Contract size | 100 oz | 1 lot = 100 ounce vàng |

### 6.2.2. Chiến lược giao dịch

Chiến lược `HybridGRUStrategy` hoạt động như sau:

**Tín hiệu:** Mô hình hybrid dự đoán nhãn cho mỗi thanh giá: Long (+1), Hold (0), hoặc Short (−1).

**Lọc tín hiệu:** Tín hiệu chỉ được thực hiện nếu:
- Độ tin cậy (confidence) vượt ngưỡng 0.50: `max(probability) > 0.50`.
- Không có vị thế đang mở (max_open_positions = 1).
- Đã qua ít nhất 6 thanh giá kể từ giao dịch trước (min_bars_between_trades = 6).

**Quản lý rủi ro:**
- Stop-loss: 2 × ATR tại thời điểm vào lệnh.
- Take-profit: 2 × ATR tại thời điểm vào lệnh.
- Max drawdown cutoff: 30% — dừng giao dịch nếu drawdown vượt ngưỡng.
- Daily loss limit: 3% — giới hạn lỗ trong 1 ngày.
- Drawdown cooldown: 12 thanh giá (12 giờ) sau khi hit max drawdown.

**Shift tín hiệu:** Tín hiệu được dịch chuyển 1 thanh giá để tránh look-ahead bias:

```python
signal = self.signals[-2]  # Dùng tín hiệu của thanh trước
```

Điều này đảm bảo quyết định giao dịch tại thanh $i$ dựa trên dự đoán tại thanh $i-1$, tương ứng với thực tế: mô hình dự đoán tại giờ $i-1$, giao dịch thực hiện tại giờ $i$.

### 6.2.3. Cảnh báo về fractional trading

`backtesting.py` không hỗ trợ giao dịch phân số (fractional trading). Vị thế được làm tròn xuống số nguyên gần nhất, có thể ảnh hưởng đến kết quả khi vốn nhỏ [2].

## 6.3. Kết quả Backtest Demo

### 6.3.1. Metrics giao dịch chính

*(Kết quả sẽ được điền sau khi chạy backtest)*

| Metric | Giá trị | Giải thích |
|---|---|---|
| Total Return | — | Lợi nhuận tổng |
| Sharpe Ratio | — | Risk-adjusted return |
| Max Drawdown | — | Mức sụt giảm vốn tối đa |
| Win Rate | — | Tỷ lệ giao dịch có lãi |
| Total Trades | — | Tổng số giao dịch |
| Profit Factor | — | Tổng lãi / Tổng lỗ |
| Avg. Trade Duration | — | Thời gian giữ vị thế trung bình |

### 6.3.2. Equity curve

Đường equity thể hiện sự thay đổi vốn theo thời gian. Đường equity lý tưởng tăng ổn định với drawdown nhỏ. Nếu equity curve có nhiều đợt sụt giảm sâu → chiến lược không ổn định.

### 6.3.3. Trade distribution

Phân tích phân phối giao dịch:
- **P&L per trade distribution:** Histogram lợi nhuận/lỗ mỗi giao dịch.
- **Win rate by direction:** Tỷ lệ thắng Long vs. Short — nếu chênh lệch lớn → mô hình thiên vị.
- **Duration distribution:** Thời gian giữ vị thế — có khớp với horizon 24 giờ không?

## 6.4. Hạn chế khi áp dụng giao dịch thực

### 6.4.1. Hạn chế của backtest

1. **Look-ahead bias tiềm ẩn:** Dù đã áp dụng nhiều biện pháp chống leakage, backtest vẫn có thể chứa bias tinh vi do:
   - Backtest replay không phản ánh độ trễ thực tế của dữ liệu real-time (latency từ feed đến lệnh thực thi).
   - Một số tham số pipeline (thông số barrier, cấu hình feature) được chọn dựa trên quan sát toàn bộ lịch sử, dù không rò rỉ trực tiếp vào model.

2. **Slippage và liquidity:** Backtest giả định thực hiện tại giá close, nhưng thực tế:
   - Lệnh lớn có thể move market (market impact).
   - Thanh khoản thấp (off-hours) → slippage lớn hơn 5 ticks.
   - Market gaps (cuối tuần, tin tức bất ngờ) → stop-loss có thể không execute tại giá mong muốn.

3. **Survivorship bias:** XAU/USD không bị delisting, nhưng các cặp tiền khác có thể bị loại bỏ — không ảnh hưởng đồ án này nhưng cần lưu ý khi tổng quát hóa.

4. **Overfitting to backtest:** Tối ưu hóa tham số dựa trên kết quả backtest → overfitting [1, 3]. Đồ án cố ý **không** tối ưu hóa tham số giao dịch.

### 6.4.2. Hạn chế khi giao dịch thực

1. **Latency:** Mô hình cần inference tại mỗi thanh giá mới. GRU + LightGBM inference nhanh (~ms) nhưng cần infrastructure ổn định.

2. **Regime change đột ngột:** Mô hình walk-forward sử dụng 2 năm dữ liệu huấn luyện — có thể không phản ánh đủ nhanh regime change (ví dụ: COVID crash tháng 3/2020).

3. **Chi phí giao dịch thực tế:** Spread, slippage, và commission trong backtest là ước tính. Giao dịch thực tế có thể đắt hơn đáng kể trong thời điểm biến động cao.

4. **Tâm lý giao dịch:** Backtest không phản ánh yếu tố tâm lý — Trader có thể can thiệp, bỏ qua tín hiệu, hoặc thay đổi chiến lược trong thời gian thực.

5. **Rủi ro mô hình:** Mô hình có thể hoạt động kém trong các điều kiện thị trường chưa từng xuất hiện trong dữ liệu huấn luyện (regime unprecedented).

### 6.4.3. Backtest không phải bằng chứng chính

López de Prado (2018) [1] cảnh báo mạnh mẽ về việc sử dụng backtest làm bằng chứng cho hiệu quả chiến lược:

- Backtest luôn có thể được tối ưu hóa (qua parameter tuning, data selection, strategy design) để tạo kết quả tốt.
- Deflated Sharpe Ratio [3] cho thấy hầu hết Sharpe ratio > 1.0 từ backtest không sống sót out-of-sample.
- Harvey et al. (2016) [4] phát hiện rằng t-statistic threshold cần > 3.0 (thay vì 1.96) để bù đắp multiple testing trong tài chính.

Vì vậy, kết quả dự báo (Chương 5) là bằng chứng chính; backtest chỉ minh họa cách mô hình **có thể** được sử dụng.

## 6.5. Tổng kết chương

Chương này đã trình bày backtest demo như một minh họa ứng dụng của mô hình hybrid, bao gồm thiết lập, chiến lược giao dịch, các metric đánh giá, và đặc biệt — các hạn chế nghiêm trọng khi áp dụng thực tế. Kết quả backtest không được coi là bằng chứng chính cho hiệu quả của phương pháp.

## Tài liệu tham khảo chương này

[1] López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

[2] backtesting.py documentation. https://kernc.github.io/backtesting.py/

[3] Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5), 94–107.

[4] Harvey, C.R., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of Expected Returns." *Review of Financial Studies*, 29(1), 5–68.
