# CHƯƠNG 5. MINH HỌA ỨNG DỤNG TÍN HIỆU

## 5.1. Vai trò của backtest

Backtest trong đồ án có vai trò minh họa cách tín hiệu Short/Hold/Long có thể được chuyển thành lệnh giao dịch giả lập. Trong nghiên cứu tài chính, backtest rất dễ bị overfit nếu nhà nghiên cứu thử nhiều tham số và chỉ chọn kết quả tốt nhất [7]. Vì vậy, đồ án không dùng backtest làm bằng chứng chính cho chất lượng mô hình.

Bằng chứng chính là kết quả classification ngoài mẫu trong walk-forward validation. Backtest chỉ trả lời câu hỏi phụ: nếu dùng tín hiệu dự báo để giao dịch theo một bộ quy tắc đơn giản, kết quả mô phỏng sẽ như thế nào?

## 5.2. Quy tắc giao dịch chi tiết

Quy tắc backtest tổng quát:

1. Nếu mô hình dự báo Long và vượt điều kiện confidence, mở hoặc giữ vị thế Long.
2. Nếu mô hình dự báo Short và vượt điều kiện confidence, mở hoặc giữ vị thế Short.
3. Nếu mô hình dự báo Hold, không vào lệnh mới.
4. Thoát lệnh theo TP/SL, tín hiệu đảo chiều hoặc điều kiện quản trị rủi ro.
5. Tính spread, slippage và commission.

### 5.2.1. Điều kiện vào lệnh (Entry conditions)

| Thành phần | Mô tả |
|---|---|
| Tín hiệu chính | Lớp có xác suất cao nhất từ mô hình (Short/Hold/Long) |
| Confidence threshold | Xác suất lớn nhất phải vượt ngưỡng (ví dụ ≥ 0.5 hoặc ≥ 0.7) |
| Entry timing | Thực thi tại giá Close của bar hiện tại hoặc giá Open của bar tiếp theo |
| Điều kiện bổ sung | Không vào lệnh mới nếu đã có vị thế đang mở cùng chiều |

Khi confidence vượt ngưỡng, hệ thống lấy tín hiệu làm hướng giao dịch. Nếu confidence dưới ngưỡng hoặc Hold là lớp có xác suất cao nhất, hệ thống không mở vị thế mới. Việc chọn threshold ảnh hưởng trực tiếp đến số lượng giao dịch và chất lượng tín hiệu: threshold thấp tạo nhiều lệnh nhưng dễ nhiễu, threshold cao giảm nhiễu nhưng bỏ lỡ cơ hội.

### 5.2.2. Quy mô vị thế (Position sizing)

| Phương án | Mô tả | Ưu điểm | Nhược điểm |
|---|---|---|---|
| Fixed lot | Mỗi lệnh dùng khối lượng không đổi (ví dụ 0.1 lot) | Đơn giản, dễ kiểm soát | Không thích ứng với biến động thị trường |
| % equity | Khối lượng tỷ lệ với vốn hiện tại (ví dụ 1–2% rủi ro mỗi lệnh) | Tự động giảm khi lỗ, tăng khi lãi | Phức tạp hơn, cần tính toán lại mỗi lệnh |
| ATR-based | Khối lượng điều chỉnh theo ATR để SL tương đương % vốn cố định | Thích ứng với biến động | Cần tính toán ATR liên tục |

Trong backtest minh họa của đồ án, phương án fixed lot được sử dụng để đơn giản hóa và tập trung vào đánh giá tín hiệu thay vì tối ưu position sizing.

### 5.2.3. Điều kiện thoát lệnh (Exit conditions)

| Loại thoát | Mô tả |
|---|---|
| Take-profit (TP) | Giá chạm mức lợi nhuận mục tiêu, chốt lời tự động |
| Stop-loss (SL) | Giá chạm mức cắt lỗ, đóng vị thế tự động |
| Tín hiệu đảo chiều | Mô hình phát tín hiệu ngược chiều (Long → Short hoặc ngược) |
| Time-based exit | Vị thế mở quá lâu (ví dụ vượt horizon label), thoát bất chấp PnL |
| Risk management exit | Đóng tất cả vị thế khi đạt giới hạn lỗ trong ngày/tuần |

TP/SL của backtest cần khớp logic label. Nếu label dùng TP/SL 2.0 ATR nhưng backtest dùng rule khác hoàn toàn, kết quả giao dịch sẽ không còn đánh giá đúng mục tiêu mà mô hình học. Time-based exit cũng quan trọng vì mô hình học trên horizon hữu hạn (24 bars), giữ vị thế quá lâu không còn nằm trong phạm vi dự báo.

### 5.2.4. Quản trị rủi ro trong giao dịch (Risk management)

| Quy tắc | Mô tả | Mục đích |
|---|---|---|
| Max concurrent positions | Giới hạn số vị thế mở đồng thời (ví dụ ≤ 2) | Tránh overexposure |
| Max daily loss | Dừng giao dịch khi tổng lỗ trong ngày vượt ngưỡng (ví dụ 2% vốn) | Bảo vệ vốn trước chuỗi lỗ |
| Max weekly loss | Dừng giao dịch tuần khi vượt ngưỡng cao hơn | Tránh drawdown sâu |
| Cooldown period | Không vào lệnh mới trong N bars sau khi thoát lệnh | Tránh overtrading do nhiễu |
| Max drawdown limit | Dừng toàn bộ hệ thống khi drawdown vượt ngưỡng | Bảo vệ vốn ở cấp hệ thống |

Các quy tắc trên chưa được triển khai đầy đủ trong backtest minh họa, nhưng được nêu để làm rõ khoảng cách giữa mô phỏng nghiên cứu và hệ thống giao dịch thực tế.

## 5.3. Kết quả minh họa mới nhất

```text
Session: results/XAUUSD_1H_20260513_023811/
Period: 2022-01-27 -> 2026-04-29
Initial equity: 10,000
Total return: 1.92%
Max drawdown: -2.72%
Sharpe ratio: 0.384
Sortino ratio: 0.637
Calmar ratio: 0.138
Profit factor: 1.109
Win rate: 47.17%
Trades: 159
```

## 5.4. Ý nghĩa kết quả

Kết quả dương nhẹ (return 1.92%, profit factor 1.109) cho thấy tín hiệu có thể được đưa qua simulator nhưng chưa chứng minh lợi thế thực tế. Sharpe 0.384 và Calmar 0.138 vẫn ở mức thấp. Max drawdown khoảng 2.72% trong mô phỏng này chưa quá lớn, nhưng không đủ để khẳng định tính ổn định.

Backtest cũng phụ thuộc mạnh vào giả định thực thi. Trong thực tế, spread của XAU/USD thay đổi theo phiên và sự kiện tin tức; slippage có thể tăng khi biến động mạnh; thanh khoản và điều kiện broker cũng ảnh hưởng kết quả. Vì vậy, mọi kết luận triển khai cần thận trọng.

## 5.5. Điều kiện triển khai thực tế

Muốn nâng cấp từ demo sang nghiên cứu triển khai, cần đáp ứng các điều kiện sau:

### 5.5.1. Hạ tầng kỹ thuật yêu cầu

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| Nguồn dữ liệu | Real-time OHLCV feed từ broker hoặc data provider, độ trễ < 1 giây |
| Máy tính | CPU đa nhân, RAM ≥ 16 GB, SSD cho lưu trữ model và log |
| Broker API | Kết nối ổn định, hỗ trợ đặt lệnh tự động, có demo account |
| Mạng | Kết nối internet ổn định, latency thấp đến broker server |
| Lưu trữ | Database cho lịch sử giao dịch, model versioning |

### 5.5.2. Xem xét độ trễ cho khung H1

Khung thời gian H1 có lợi thế hơn so với M1 hay M5 về độ trễ thực thi. Mỗi bar kéo dài 60 phút, nên độ trễ vài giây đến vài chục giây giữa lúc tín hiệu xuất hiện và lúc lệnh được thực thi thường không ảnh hưởng đáng kể đến giá entry. Tuy nhiên, trong các thời điểm biến động mạnh (tin tức kinh tế quan trọng, sự kiện địa chính trị), spread có thể giãn rộng đột ngột và slippage tăng đáng kể ngay trên khung H1. Do đó, hệ thống cần có cơ chế kiểm tra spread trước khi vào lệnh và tự động bỏ qua tín hiệu khi spread vượt ngưỡng cho phép.

### 5.5.3. Giai đoạn paper trading

Trước khi giao dịch thật, cần trải qua giai đoạn paper trading tối thiểu 3–6 tháng:

- Chạy hệ thống trên tài khoản demo với điều kiện thị trường thật.
- Ghi nhận mọi tín hiệu, lệnh giả lập và so sánh với backtest.
- Đo lường độ lệch giữa kết quả paper và backtest (slippage thực tế, spread thực tế, requote).
- Xác nhận hệ thống hoạt động ổn định: không crash, không mất tín hiệu, log đầy đủ.
- Chỉ tiến lên giao dịch thật nếu paper trading cho thấy kết quả nhất quán với kỳ vọng từ backtest.

### 5.5.4. Giám sát mô hình

| Hạng mục giám sát | Phương pháp | Tần suất |
|---|---|---|
| Model drift | So sánh phân phối xác suất dự báo qua từng tháng | Hàng tuần |
| Regime detection | Theo dõi volatility cluster, ATR trend, correlation thay đổi | Hàng ngày |
| Performance decay | Tracking rolling accuracy, F1, directional accuracy | Hàng tuần |
| Data quality | Kiểm tra missing bars, giá bất thường, volume bất thường | Hàng ngày |
| Retrain trigger | Tự động đề xuất retrain khi accuracy giảm quá ngưỡng | Khi cần |

Mô hình học máy tài chính không thể hoạt động ổn định mãi mãi do non-stationarity của dữ liệu thị trường. Cần có quy trình retrain định kỳ (ví dụ hàng quý) và quy trình retrain khẩn cấp khi phát hiện regime shift.

### 5.5.5. Yếu tố pháp lý và quản trị

- Giao dịch XAU/USD thường thực hiện qua CFD (Contract for Difference), cần hiểu rõ cơ chế margin, leverage và rủi ro đòn bẩy.
- Mỗi khu vực pháp lý có quy định khác nhau về giao dịch CFD: giới hạn leverage, yêu cầu vốn tối thiểu, quy trình KYC/AML.
- Broker yêu cầu ký thỏa thuận rủi ro và có thể áp dụng margin call hoặc tự động đóng vị thế.
- Cần tuân thủ quy định về báo cáo thuế từ hoạt động giao dịch tài chính theo pháp luật sở tại.
- Nếu triển khai institutional, cần tuân thủ thêm các quy định về best execution, record keeping và risk disclosure.

## 5.6. Giới hạn backtest minh họa

Backtest trong đồ án chịu nhiều giới hạn cần nêu rõ để tránh diễn giải sai:

### 5.6.1. Giả định về spread

Spread trong backtest được giả định cố định hoặc dựa trên giá trị trung bình lịch sử. Trong thực tế, spread của XAU/USD thay đổi đáng kể theo phiên giao dịch: spread thường hẹp trong phiên London và New York, nhưng giãn rộng trong phiên Asian. Spread cũng tăng mạnh trước, trong và sau các sự kiện tin tức quan trọng (NFP, FOMC, CPI). Nếu backtest không mô phỏng spread thay đổi, kết quả có thể lạc quan, đặc biệt đối với các lệnh giao dịch trong phiên thanh khoản thấp.

### 5.6.2. Slippage trong sự kiện tin tức

Backtest không mô phỏng slippage do tin tức kinh tế trọng đại. Trong thực tế, khi dữ liệu NFP, FOMC rate decision hoặc CPI được công bố, giá XAU/USD có thể nhảy (gap) vài chục pip trong vài giây. Lệnh đặt trước tin tức có thể bị fill ở giá rất khác so với giá mong muốn. Backtest giả định thực thi tại giá Close hoặc Open của bar, bỏ qua hiện tượng này hoàn toàn.

### 5.6.3. Không mô phỏng market impact

Backtest giả định mỗi lệnh giao dịch không ảnh hưởng đến giá thị trường. Giả định này hợp lý với khối lượng nhỏ, nhưng nếu mở rộng lên quy mô lớn hơn, lệnh giao dịch có thể di chuyển giá bất lợi (price impact). Đặc biệt với tài sản CFD, thanh khoản phụ thuộc vào broker và có thể bị hạn chế trong điều kiện thị trường bất thường.

### 5.6.4. Không có quản trị rủi ro cấp portfolio

Backtest chỉ quản lý một tài sản (XAU/USD) trên một khung thời gian (H1). Không có cơ chế quản trị rủi ro ở cấp portfolio như: phân bổ vốn giữa nhiều tài sản, hedge tương quan, hoặc điều chỉnh exposure theo trạng thái thị trường tổng thể. Một hệ thống giao dịch thực tế cần quản lý nhiều rủi ro đồng thời, không chỉ rủi ro của từng vị thế đơn lẻ.

### 5.6.5. Đơn tài sản, đơn khung thời gian

Toàn bộ backtest chỉ xét XAU/USD trên H1. Không kiểm tra tương quan với các tài sản khác (DXY, US10Y, EUR/USD), không kiểm tra tín hiệu trên nhiều khung thời gian (multi-timeframe analysis), và không kiểm tra tính bền vững của chiến lược trên các tài sản khác. Mở rộng sang multi-asset và multi-timeframe là hướng phát triển quan trọng nhưng nằm ngoài phạm vi đồ án.

## 5.7. Kết luận chương

Backtest minh họa hoàn thành vai trò chứng minh pipeline có thể biến dự báo thành hành động giao dịch giả lập. Tuy nhiên, kết quả hiện tại chưa đủ để khẳng định chiến lược sinh lời. Luận văn nên trình bày backtest như phần ứng dụng phụ, còn trọng tâm học thuật nằm ở quy trình labeling, validation, model comparison và phân tích lỗi.

## 5.8. Thiết kế tín hiệu giao dịch từ xác suất mô hình

Mô hình classification không trực tiếp xuất lệnh giao dịch; nó xuất xác suất cho các lớp. Một lớp chuyển đổi tín hiệu cần quyết định:

```text
Nếu P(Long) lớn nhất và confidence đủ cao -> Long
Nếu P(Short) lớn nhất và confidence đủ cao -> Short
Nếu P(Hold) lớn nhất hoặc confidence thấp -> Không vào lệnh
```

Confidence có thể được định nghĩa là xác suất lớn nhất trong ba lớp:

```text
confidence = max(P(Short), P(Hold), P(Long))
```

Nếu threshold quá thấp, hệ thống giao dịch nhiều và dễ nhiễu. Nếu threshold quá cao, hệ thống giao dịch ít, có thể bỏ lỡ cơ hội. Kết quả high-confidence ở Chương 4 cho thấy threshold 0.7 tạo rất ít mẫu, vì vậy cần calibration trước khi dùng threshold như một quyết định thực tế.

## 5.9. Chi phí giao dịch và giả định thực thi

Backtest chỉ có ý nghĩa khi giả định chi phí được nêu rõ. Với XAU/USD CFD, chi phí có thể gồm:

- Spread giữa bid và ask.
- Commission theo lot.
- Slippage khi thị trường biến động mạnh.
- Swap/overnight fee nếu giữ qua ngày.
- Giới hạn margin và leverage.
- Khác biệt giá giữa broker và nguồn dữ liệu.

Nếu bỏ qua chi phí, backtest thường lạc quan. Trong kết quả hiện tại, profit factor chỉ vừa trên 1 và Sharpe vẫn thấp, vì vậy chỉ cần chi phí thực tế tăng nhẹ cũng có thể làm chiến lược xấu đi.

## 5.10. Quản trị rủi ro tối thiểu

Một ứng dụng thực tế không nên chỉ dựa vào tín hiệu mô hình. Cần thêm quản trị rủi ro:

| Thành phần | Vai trò |
|---|---|
| Max position size | Giới hạn rủi ro mỗi lệnh |
| Daily loss limit | Dừng giao dịch khi lỗ trong ngày vượt ngưỡng |
| Cooldown | Tránh vào lệnh liên tục sau tín hiệu nhiễu |
| Volatility filter | Giảm giao dịch khi biến động bất thường |
| Session filter | Chỉ giao dịch phiên có thanh khoản phù hợp |
| Confidence threshold | Chỉ vào lệnh khi xác suất đủ rõ |

Các thành phần này chưa phải trọng tâm đồ án, nhưng cần nêu để tránh hiểu nhầm rằng mô hình classification là đủ cho hệ thống giao dịch thực.

## 5.11. Phân biệt nghiên cứu và triển khai

Bảng sau phân biệt phạm vi của luận văn và yêu cầu triển khai thật:

| Hạng mục | Trong luận văn | Triển khai thật |
|---|---|---|
| Dữ liệu | Historical OHLCV | Realtime feed ổn định |
| Mô hình | Train offline | Retrain/monitor định kỳ |
| Đánh giá | Walk-forward + backtest demo | Paper trading + live monitoring |
| Chi phí | Giả định mô phỏng | Chi phí broker thực tế |
| Rủi ro | Mô tả cơ bản | Risk engine độc lập |
| Vận hành | Script/pipeline | Hệ thống giám sát lỗi |

Vì vậy, kết luận của Chương 6 chỉ nên nói: pipeline có thể minh họa cách tín hiệu được dùng trong giao dịch giả lập. Không nên nói: hệ thống đã sẵn sàng giao dịch thật.

## 5.12. Kịch bản cải thiện backtest

Nếu tiếp tục nghiên cứu, các thí nghiệm hợp lý gồm:

1. Dùng confidence threshold sau khi calibration.
2. Chỉ giao dịch khi LightGBM và Stacking đồng thuận.
3. Lọc theo phiên London/New York.
4. Lọc theo volatility regime.
5. Tối ưu position sizing ngoài tập test.
6. Kiểm tra sensitivity với spread/slippage.
7. Tách riêng giai đoạn validation cho rule giao dịch, không dùng test set để chọn threshold.

Điểm quan trọng là mọi cải thiện backtest phải có validation riêng. Nếu chọn rule dựa trên cùng kết quả backtest đã báo cáo, nguy cơ overfit rất cao.

## 5.13. Kết luận mở rộng chương ứng dụng

Chương ứng dụng cho thấy khoảng cách giữa mô hình dự báo và hệ thống giao dịch. Một mô hình có thể có metric classification tốt nhưng backtest kém do chi phí, timing và risk management. Ngược lại, một backtest đẹp cũng không đủ nếu classification evaluation bị leakage. Vì vậy luận văn đặt trọng tâm vào quy trình học máy trước, rồi dùng backtest như minh họa có kiểm soát.
