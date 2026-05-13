# CHƯƠNG 4. THỰC NGHIỆM VÀ KẾT QUẢ

## 4.1. Thiết lập thực nghiệm

Thiết lập chính:

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | XAU/USD H1 |
| Giai đoạn test/backtest | 2022-01-27 đến 2026-04-29 |
| Nhãn | Short/Hold/Long theo triple-barrier |
| TP/SL labeling | 2.0 ATR / 2.0 ATR |
| Horizon | 24 bars H1 |
| Validation | Walk-forward có purge/embargo |
| Runtime chính | Classic Hybrid Stacking |
| Phiên kết quả | `results/XAUUSD_1H_20260513_023811/` |

Các mô hình được so sánh:

1. Naive Direction baseline.
2. Majority baseline.
3. Random baseline.
4. Logistic Regression.
5. Random Forest.
6. LightGBM.
7. Hybrid Stacking.

## 4.2. Phân phối nhãn

Phân phối nhãn trong tập đánh giá:

```text
Short: 43.0%
Hold :  8.5%
Long : 48.5%
```

Lớp Hold thấp hơn mức lý tưởng. Tuy nhiên, thử `horizon_bars = 48` làm Hold giảm còn khoảng 1.5%, do đó cấu hình 48 bị loại. Điều này cho thấy trên dữ liệu XAU/USD H1, barrier 2.0 ATR thường bị chạm trước khi hết horizon. Đây là đặc điểm cần được thảo luận thay vì che giấu.

Mất cân bằng lớp ảnh hưởng đến cách diễn giải metrics. Accuracy có thể bị chi phối bởi Short/Long, trong khi Macro F1 phản ánh tốt hơn hiệu quả trên cả ba lớp.

## 4.3. Kết quả Hybrid Stacking

Kết quả chính:

```text
Total samples         23,910
Accuracy              0.3416
Balanced Accuracy     0.3675
Directional Accuracy  0.4929
Directional Baseline  0.5000
Majority Baseline     0.4850
Macro F1              0.3152
Weighted F1           0.3674
```

Per-class metrics:

| Lớp | True count | Pred count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Short | 10,280 | 7,374 | 0.436 | 0.313 | 0.364 |
| Hold | 2,033 | 8,013 | 0.112 | 0.440 | 0.178 |
| Long | 11,597 | 8,523 | 0.476 | 0.350 | 0.404 |

Nhận xét:

- Mô hình có F1 tốt hơn ở Short và Long so với Hold.
- Hold có precision thấp vì mô hình dự báo Hold nhiều hơn số Hold thật.
- Directional Accuracy 0.4929 gần baseline 0.5, cho thấy tín hiệu hướng chưa đủ mạnh.
- Macro F1 0.3152 phản ánh bài toán ba lớp còn khó và chưa đạt mức triển khai thực tế.

## 4.4. So sánh mô hình toàn diện

Bảng tổng hợp tất cả mô hình, bao gồm cả Balanced Accuracy và Weighted F1 để đánh giá cân bằng lớp:

| Mô hình | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Dir. Acc | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| Naive Direction | 0.4574 | — | 0.3178 | — | 0.5000 | Baseline hướng |
| Majority | 0.4850 | — | 0.2177 | — | 0.5301 | Dự báo lớp đa số |
| Random | 0.3361 | — | 0.3056 | — | 0.5000 | Baseline ngẫu nhiên |
| Logistic Regression | 0.3568 | 0.373 | 0.3173 | 0.362 | — | Base model tuyến tính |
| Random Forest | 0.3596 | 0.387 | 0.3280 | 0.375 | — | Bagging ensemble |
| LightGBM | 0.3738 | 0.364 | 0.3265 | 0.380 | — | Gradient boosting |
| Hybrid Stacking | 0.3416 | 0.368 | 0.3152 | 0.367 | 0.4929 | Runtime chính |

Ghi chú: Balanced Accuracy được tính bằng trung bình recall của ba lớp; Weighted F1 được tính theo support từng lớp. Baseline không có per-class metrics nên để trống.

Per-class F1 chi tiết (Short/Hold/Long):

| Mô hình | Short F1 | Hold F1 | Long F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.466 | 0.186 | 0.300 |
| Random Forest | 0.451 | 0.193 | 0.340 |
| LightGBM | 0.470 | 0.173 | 0.336 |
| Hybrid Stacking | 0.364 | 0.178 | 0.404 |

Nhận xét tổng hợp:

1. **Accuracy cao nhất thuộc về Majority Baseline (0.4850)** — nhưng đây là ảo giác do mất cân bằng lớp. Macro F1 của Majority chỉ 0.2177, thấp nhất trong tất cả mô hình.
2. **LightGBM đạt Accuracy (0.3738) và Weighted F1 (0.380) cao nhất** trong các mô hình học máy, cho thấy đây là base learner mạnh nhất.
3. **Random Forest có Balanced Accuracy cao nhất (0.387)** nhờ recall Hold tốt nhất (0.447).
4. **Hybrid Stacking có Long F1 cao nhất (0.404)** nhưng Short F1 thấp hơn cả ba base models, kéo Macro F1 tổng thể xuống.
5. **Hold F1 luôn thấp (0.173–0.193)** ở tất cả mô hình — đây là điểm yếu chung không chỉ của Hybrid Stacking.

LightGBM đơn lẻ đạt Accuracy và Macro F1 cao hơn Hybrid Stacking trong lần chạy này. Đây không nên được xem là lỗi triển khai. Theo nguyên lý ensemble, stacking có thể cải thiện khi các base learners có lỗi bổ sung cho nhau; nhưng nếu base models học tín hiệu yếu hoặc lỗi tương quan cao, meta-model không nhất thiết vượt mô hình tốt nhất [18]. Với dữ liệu tài chính nhiễu cao, tăng độ phức tạp còn có thể làm tăng rủi ro overfit [7].

## 4.5. Confusion matrix

Confusion matrix ba lớp:

| True \ Pred | Short | Hold | Long |
|---|---:|---:|---:|
| Short | 3,213 | 3,234 | 3,833 |
| Hold | 510 | 894 | 629 |
| Long | 3,651 | 3,885 | 4,061 |

Kết quả cho thấy mô hình nhầm khá nhiều giữa Short và Long, đồng thời cũng đẩy nhiều mẫu directional sang Hold. Điều này phù hợp với đặc thù dữ liệu tài chính nhiễu cao: tín hiệu kỹ thuật không ổn định qua mọi regime.

Các điểm chính:

1. True Short bị chia gần đều sang Short/Hold/Long, cho thấy mô hình chưa phân biệt tốt điều kiện giảm.
2. True Long cũng bị chia mạnh sang Short/Hold/Long, thể hiện tín hiệu hướng chưa ổn định.
3. Hold recall đạt 0.440 nhưng precision chỉ 0.112, nghĩa là mô hình dự báo Hold quá rộng so với số Hold thật.
4. Sai nhầm Short thành Long (3,833 mẫu) và Long thành Short (3,651 mẫu) là lỗi nghiêm trọng nhất nếu dùng giao dịch — đi ngược hướng thị trường.
5. Sai nhầm directional thành Hold chủ yếu là bỏ lỡ cơ hội, ít nguy hiểm hơn vào sai hướng.

## 4.6. Phân tích theo walk-forward window

Walk-forward validation sử dụng 25 cửa sổ trượt, mỗi cửa sổ test gồm 990 bars (~1 tháng). Accuracy dao động mạnh giữa các cửa sổ:

| Cửa sổ | Giai đoạn test (xấp xỉ) | Accuracy | Nhận xét |
|---|---|---:|---|
| 1 | 2022-01 → 2022-03 | 0.331 | Đầu năm, vàng dao động quanh $1.800–1.950 |
| 4 | 2022-07 → 2022-09 | 0.239 | Fed tăng lãi suất mạnh, vàng giảm |
| 8 | 2023-04 → 2023-06 | 0.409 | Khủng hoảng ngân hàng Mỹ, vàng tăng trend |
| 9 | 2023-06 → 2023-08 | **0.488** | Cao nhất — xu hướng rõ |
| 11 | 2023-10 → 2023-12 | **0.201** | Thấp nhất — thị trường sideways |
| 16 | 2024-08 → 2024-10 | 0.459 | Fed pivot kỳ vọng, trend rõ |
| 19 | 2025-02 → 2025-04 | 0.210 | Sideways, nhiễu cao |
| 25 | 2026-02 → 2026-04 | 0.467 | Cửa sổ cuối, trend khá |

Thống kê tổng quan:

```text
Số cửa sổ:        25
Accuracy trung bình: 0.346
Accuracy cao nhất:   0.488 (Cửa sổ 9)
Accuracy thấp nhất:  0.201 (Cửa sổ 11)
Độ lệch chuẩn:      ≈ 0.075
Cửa sổ > 0.40:      8/25 (32%)
Cửa sổ < 0.25:      3/25 (12%)
```

Quan sát chính:

1. **Hiệu suất biến động lớn giữa các cửa sổ** — từ 0.201 đến 0.488 — cho thấy mô hình nhạy cảm với regime thị trường.
2. **Cửa sổ có xu hướng rõ (trending)** thường cho accuracy cao hơn (W8, W9, W16, W25). Khi giá vàng có trend rõ ràng, barrier TP/SL dễ phân biệt hơn.
3. **Cửa sổ sideways / nhiễu cao** (W4, W11, W19) có accuracy thấp nhất. Trong giai đoạn này, giá chạm barrier ngẫu nhiên, nhãn Hold tăng, mô hình khó phân biệt.
4. **Không có xu hướng suy giảm (degradation)** theo thời gian — accuracy không giảm dần từ W1 đến W25, cho thấy mô hình không bị overfit vào giai đoạn đầu.

Hình 4.1 minh họa hiệu suất mô hình theo từng cửa sổ walk-forward.

## 4.7. Phân tích lỗi và thảo luận

### 4.7.1. Tại sao Hold có precision thấp?

Hold chỉ chiếm 8.5% tập test (2,033/23,910 mẫu) nhưng mô hình dự báo Hold tới 8,013 lần — tức dự báo Hold nhiều gấp gần 4 lần số lượng Hold thật. Kết quả là precision chỉ đạt 0.112.

Nguyên nhân sâu xa:

1. **Mất cân bằng lớp nghiêm trọng.** Hold chỉ có ~8.5% trong khi Short (43.0%) và Long (48.5%) chiếm đa số. Mô hình học được rằng "khi không chắc chắn, dự báo Hold an toàn hơn" — đây là hiện tượng bias phổ biến khi lớp thiểu số không đủ đại diện.
2. **Thiết kế TP/SL và horizon.** Như đã phân tích trong §2.13 (Chương 2), triple-barrier với `horizon = 24 bars` và `ATR mult = 2.0` khiến hầu hết nến chạm TP hoặc SL trước khi hết horizon. Khi giá dao động trong biên độ hẹp, cả hai barrier đều chưa bị chạm khi hết horizon → nhãn Hold. Nhưng ranh giới giữa "gần chạm" và "chưa chạm" rất mỏng, khiến mô hình nhầm Hold với Short/Long.
3. **Feature không phân biệt rõ Hold.** Các feature kỹ thuật (RSI, EMA cross, VWAP…) phản ánh momentum và trend, vốn có khả năng phân biệt Short vs Long tốt hơn là "không đi về đâu". Hold bản chất là trạng thái thiếu tín hiệu, khó biểu diễn bằng feature.

Hệ quả thực tế: precision Hold thấp làm giảm tỷ lệ giao dịch đúng khi theo tín hiệu Hold. Tuy nhiên, nếu chỉ dùng Short/Long (bỏ Hold), bài toán trở thành nhị phân và accuracy có thể cải thiện — nhưng mất khả năng lọc tín hiệu yếu.

### 4.7.2. Tại sao Hybrid Stacking không vượt LightGBM?

Stacking đạt Accuracy 0.3416 so với LightGBM 0.3738 — chênh lệch -3.2 điểm phần trăm. Macro F1 cũng thấp hơn (0.3152 vs 0.3265). Có nhiều nguyên nhân giải thích:

1. **Base learners có lỗi tương quan cao.** Stacking cải thiện khi các base models sai khác nhau (errors are complementary) [18]. Tuy nhiên, Logistic Regression, Random Forest và LightGBM đều học từ cùng 25 feature trên cùng dữ liệu XAU/USD — chúng sai ở những nơi tương tự nhau. Meta-learner không có thêm thông tin để sửa lỗi.
2. **Meta-split làm giảm dữ liệu huấn luyện.** Base models chỉ dùng 80% train data (≈4,990 mẫu), meta-learner dùng 20% còn lại (≈1,248 mẫu). LightGBM đơn lẻ được train trên toàn bộ train data mỗi window, nên tận dụng dữ liệu tốt hơn.
3. **Tín hiệu yếu trong dữ liệu tài chính.** Thị trường vàng có tính hiệu quả cao (EMH) [1], nên tín hiệu dự báo vốn đã yếu. Khi signal-to-noise ratio thấp, tăng độ phức tạp mô hình (từ 1 model lên 4 models) không tạo thêm tín hiệu — chỉ tăng nhiễu và rủi ro overfit [7].
4. **Meta-learner có thể overfit vào pattern của base learners.** Logistic Regression làm meta-learner học từ 9 feature (3 probability × 3 models), nhưng 1,248 mẫu train cho meta-learner có thể không đủ để học mối quan hệ phức tạp giữa base predictions và label thật. Meta-learner có xu hướng học shortcut — ghi nhớ pattern của base learners thay vì signal thực sự.

Liên hệ bias-variance (§2.7, Chương 2): LightGBM đơn lẻ có bias thấp và variance kiểm soát tốt nhờ boosting. Stacking thêm Random Forest (variance cao) và Logistic Regression (bias cao), trung bình không cải thiện mà có thể làm tăng tổng sai số.

### 4.7.3. Directional Accuracy gần baseline ngẫu nhiên

Directional Accuracy của Hybrid Stacking là 0.4929, thấp hơn baseline ngẫu nhiên (0.5000). Điều này có nghĩa là nếu chỉ xem xét hướng dự báo (Short/Long), mô hình dự báo đúng hướng chưa bằng tung đồng xu.

Diễn giải:

1. **Kết quả này phù hợp với giả thuyết thị trường hiệu quả (EMH)** [1]. Nếu thị trường thực sự hiệu quả ở dạng weak, thì giá lịch sử (OHLCV và feature kỹ thuật) không chứa tín hiệu đủ mạnh để dự báo hướng giá tiếp theo một cách đáng tin cậy.
2. **Directional Accuracy tổng thể bị kéo xuống bởi các cửa sổ nhiễu.** Như phân tích trong §4.6, accuracy dao động từ 0.201 đến 0.488. Trong các cửa sổ sideways, mô hình dự báo hướng sai gần ngẫu nhiên, kéo Dir. Acc. tổng thể về 0.5.
3. **Dir. Acc. cao hơn ở high-confidence (0.559)** nhưng chỉ có 182 mẫu (0.76%) — chưa đủ kiểm định thống kê.
4. **Hệ quả cho giao dịch.** Dir. Acc. < 0.5 tương đương với việc theo tín hiệu mô hình sẽ lỗ sau chi phí giao dịch. Tuy nhiên backtest demo (§4.9) vẫn dương vì profit factor > 1 khi đúng hướng bù đắp sai hướng — tức các trade đúng có PnL lớn hơn các trade sai.

Hình 4.2 so sánh Directional Accuracy của Hybrid Stacking với các baseline.

## 4.8. Kết quả high-confidence

Với threshold confidence 0.7:

```text
Số mẫu high-confidence: 182
Tỷ lệ trên tổng mẫu: 0.76%
Accuracy: 0.242
Directional Accuracy: 0.559
```

Directional Accuracy high-confidence cao hơn tổng thể, nhưng số mẫu chỉ khoảng 0.76%, chưa đủ để kết luận chắc chắn. Đây là hướng có thể phát triển thêm: calibration và thresholding theo confidence.

Phân tích sâu hơn:

- **Accuracy thấp (0.242)** dù Dir. Acc. cao hơn (0.559) cho thấy mô hình self-confident nhưng sai nhiều ở lớp Hold. Khi confidence cao, mô hình thiên về dự báo Hold hoặc Long/Short cực đoan.
- **Calibration:** ECE = 0.0846 cho thấy mô hình moderately calibrated — confidence score khá tương đồng với xác suất thực tế, nhưng vẫn over/under-confident ở một số bin. Brier score = 0.2247, Log-loss = 1.1047.
- Hướng phát triển: calibration isotonic regression và threshold tuning, nhưng phải tránh overfit threshold trên cùng test set.

## 4.9. Backtest minh họa

Backtest demo cho kết quả:

```text
Initial equity  10,000
Total return    1.92%
Max drawdown   -2.72%
Sharpe ratio    0.384
Sortino ratio   0.637
Calmar ratio    0.138
Profit factor   1.109
Win rate        47.17%
Trades          159
Khoảng thời gian 2022-01-27 → 2026-04-29
```

Đánh giá theo thang chất lượng (xem thêm Boring Edge [backtest-metrics-explained]):

| Metric | Giá trị | Vùng | Khuyến nghị |
|---|---:|---|---|
| Total Return | 1.92% | 0–10% — Thấp | > 10% |
| Sharpe Ratio | 0.384 | 0–1.0 — Chấp nhận | > 1.0 |
| Max Drawdown | 2.72% | < 15% — Tốt | < 15% |
| Win Rate | 47.17% | 40–55% — Chấp nhận | > 55% |
| Profit Factor | 1.109 | 1.0–1.5 — Biên | > 1.5 |
| Calmar Ratio | 0.138 | 0–1.0 — Chấp nhận | > 1.0 |
| Sortino Ratio | 0.637 | 0–1.0 — Chấp nhận | > 1.0 |
| Avg Win/Avg Loss | 1.35 | 1.0–1.5 — Biên | > 1.5 |
| Expectancy | ~0% | 0–0.5% — Nhỏ | > 0.5% |

Kết quả có return dương nhẹ, profit factor trên 1 và Sharpe thấp nhưng dương. Max drawdown được kiểm soát tốt (2.72%). Tuy nhiên Sharpe 0.384 vẫn ở mức chưa đủ mạnh để kết luận chiến lược có lợi thế giao dịch thực tế. Kết quả này chỉ chứng minh pipeline có thể chuyển tín hiệu ML thành giao dịch giả lập và đo lường rủi ro/lợi nhuận.

So sánh với benchmark:

| Chiến lược | Return | Sharpe | Max DD | Ghi chú |
|---|---:|---:|---:|---|
| Buy & Hold | 153.2% | 1.34 | 24.4% | Không chi phí giao dịch |
| Always Long (10x) | 1,677.3% | 1.34 | 98.6% | Rủi ro cực cao |
| Random Signal | -99.7% | 0.15 | 100.0% | Phá tài |
| Hybrid Stacking | 1.9% | 0.38 | 2.7% | Có chi phí giao dịch |

Hybrid Stacking có return thấp nhất nhưng max drawdown được kiểm soát tốt nhất. Buy & Hold và Always Long có return cao hơn nhiều nhưng rủi ro cũng cao hơn tương ứng — Always Long mất 98.6% vốn ở đâu đó trong giai đoạn test.

## 4.10. Diễn giải kết quả

Các kết luận chính:

1. Pipeline chạy được đầy đủ từ dữ liệu đến báo cáo.
2. Labeling và validation đã được thiết kế theo chuẩn phù hợp hơn cho dữ liệu tài chính.
3. LightGBM là baseline mạnh nhất trong lần chạy hiện tại.
4. Hybrid Stacking không vượt LightGBM, cho thấy ensemble phức tạp không đảm bảo cải thiện ngoài mẫu.
5. Lớp Hold là điểm yếu lớn nhất do phân phối thấp và precision kém.
6. Backtest demo cho kết quả dương nhẹ nhưng chưa đủ mạnh để khẳng định profitability.

Kết quả này vẫn có giá trị học thuật vì nó minh họa đầy đủ quy trình kiểm định có kiểm soát. Trong tài chính, một kết quả trung thực cho thấy mô hình chưa vượt baseline cũng quan trọng vì nó tránh overclaim và phù hợp với cảnh báo về backtest overfitting [7], [9].

## 4.11. Hạn chế thực nghiệm

- Chỉ sử dụng OHLCV và feature kỹ thuật, chưa có dữ liệu vĩ mô/tin tức/sentiment.
- Hold class thấp, làm bài toán ba lớp khó cân bằng.
- Chưa có calibration xác suất (isotonic/platt).
- Chưa kiểm tra robustness qua nhiều cấu hình chi phí giao dịch (spread/slippage khác nhau).
- Chưa phân tích SHAP theo từng regime.
- Chưa có out-of-time test trên dữ liệu sau 2026-04-30.
- Walk-forward chỉ dùng sliding window; chưa thử anchored hoặc expanding window.

## 4.12. Môi trường thực nghiệm chi tiết

Môi trường thực nghiệm:

| Thành phần | Vai trò |
|---|---|
| Python | Ngôn ngữ triển khai pipeline |
| Pixi | Quản lý môi trường và command workflow |
| Polars | Xử lý dữ liệu dạng bảng/parquet hiệu năng cao |
| NumPy | Tính toán numeric |
| scikit-learn | Logistic Regression, Random Forest, metrics |
| LightGBM | Gradient boosting tree |
| Ruff | Kiểm tra style/lint |
| compileall | Kiểm tra lỗi cú pháp Python |

Các bước validation đã chạy trước khi viết báo cáo:

```text
pixi run ruff check src
pixi run python -m compileall -q src tests
pixi run python main.py --stage 2 --force
```

Stage 2 trở đi đã tạo session kết quả mới nhất dùng trong báo cáo.

## 4.13. Artifact thực nghiệm

Các artifact chính:

| File | Nội dung |
|---|---|
| `reports/model_metrics.json` | Metrics tổng, per-class, confusion matrix |
| `reports/model_comparison.md` | So sánh baseline/base models/stacking |
| `reports/feature_importance.json` | Feature importance sau training |
| `reports/walk_forward_history.json` | Lịch sử từng cửa sổ walk-forward |
| `backtest/backtest_results.json` | Kết quả backtest demo |
| `backtest/backtest_chart.html` | Biểu đồ backtest |
| `backtest/trades_detail.csv` | Chi tiết từng giao dịch |
| `backtest/equity_curve.csv` | Đường equity theo thời gian |

Việc lưu artifact giúp báo cáo không chỉ dựa trên mô tả miệng. Mỗi số liệu trong chương này có thể kiểm tra lại từ file kết quả.

## 4.14. So sánh với baseline đa số

Majority Baseline có accuracy 0.4850 nhưng macro F1 chỉ 0.2177. Điều này minh họa vì sao accuracy không đủ. Dự báo lớp phổ biến nhất có thể đạt accuracy cao hơn Hybrid Stacking, nhưng không nhận diện cân bằng cả ba lớp. Macro F1 của Hybrid Stacking cao hơn Majority Baseline (0.3152 vs 0.2177), dù accuracy thấp hơn.

```text
Accuracy của Majority Baseline cao do phân phối lớp lệch. Tuy nhiên Macro F1 thấp cho thấy baseline này bỏ qua các lớp còn lại. Vì vậy báo cáo ưu tiên Macro F1 và per-class F1 khi đánh giá mô hình ba lớp.
```

## 4.15. So sánh với LightGBM

LightGBM đạt:

```text
Accuracy  0.3738
Macro F1  0.3265
Short F1  0.470
Long F1   0.336
Hold F1   0.173
```

Hybrid Stacking đạt:

```text
Accuracy  0.3416
Macro F1  0.3152
Long F1   0.404
Short F1  0.364
Hold F1   0.178
```

LightGBM tốt hơn về accuracy (+3.2pp) và macro F1 (+1.1pp). Hybrid Stacking có Long F1 cao hơn (0.404 vs 0.336) nhưng Short F1 thấp hơn đáng kể (0.364 vs 0.470). Hold F1 gần tương đương (0.178 vs 0.173).

Điều này cho thấy stacking không hoàn toàn vô ích — nó học được Long signal tốt hơn — nhưng meta-model chưa tạo cải thiện tổng thể. Có thể lỗi giữa base learners tương quan cao, hoặc meta split làm giảm lượng dữ liệu train base models. Phân tích chi tiết hơn trong §4.7.2.

## 4.16. High-confidence analysis

High-confidence threshold 0.7 chỉ tạo 182 mẫu, chiếm 0.76% tổng số mẫu. Directional Accuracy tăng lên 0.559 nhưng mẫu quá ít. Có hai cách diễn giải:

- **Tích cực:** khi mô hình tự tin, tín hiệu hướng có vẻ tốt hơn. Điều này phù hợp với intuition — high confidence phản ánh feature pattern rõ ràng hơn.
- **Thận trọng:** số mẫu quá ít (182/23,910) nên kết luận chưa chắc chắn, cần thêm kiểm định thống kê (ví dụ permutation test).

Calibration: ECE = 0.0846 (moderate), Brier = 0.2247, Log-loss = 1.1047. Mô hình moderately calibrated — confidence khá tương đồng với thực tế.

Hướng phát triển là calibration và threshold tuning, nhưng phải tránh overfit threshold trên cùng test set.

## 4.17. Feature importance sau pruning

Top-10 feature theo importance:

| Hạng | Feature | Nhóm | Điểm |
|---|---|---|---:|
| 1 | `vwap` | Kỹ thuật | 29 |
| 2 | `ema34_vs_ema89` | Kỹ thuật | 24 |
| 3 | `adx_14` | Kỹ thuật | 16 |
| 4 | `atr_pct_close` | Kỹ thuật | 14 |
| 5 | `atr_percentile` | Kỹ thuật | 10 |
| 6 | `pivot_position` | Kỹ thuật | 9 |
| 7 | `price_dist_ratio` | Kỹ thuật | 7 |
| 8 | `atr_ratio` | Kỹ thuật | 6 |
| 9 | `price_position_20` | Kỹ thuật | 6 |
| 10 | `rsi_14` | Kỹ thuật | 5 |

Sau khi bỏ 4 feature importance thấp, feature set còn 21 feature. Mục tiêu không phải tạo kết quả tốt hơn ngay lập tức, mà là giảm nhiễu và giúp mô hình dễ giải thích. Báo cáo trình bày feature pruning như một quyết định kỹ thuật có kiểm soát:

1. Dựa trên kết quả feature importance thật.
2. Chỉ bỏ feature ít đóng góp.
3. Không thêm mô hình mới khi label/feature chưa ổn.
4. Rerun pipeline để kiểm chứng.

Quan sát: tất cả top-10 feature đều là feature kỹ thuật (0% model-derived), cho thấy pipeline hiện chỉ dựa vào price-action.

## 4.18. Phân tích backtest demo

Backtest có 159 giao dịch, win rate 47.17%, return 1.92%, profit factor 1.109. Đây là kết quả dương nhẹ nhưng Sharpe 0.384 vẫn ở mức thấp.

Một chiến lược thực tế cần:

- Profit factor lớn hơn 1 một cách ổn định.
- Sharpe đủ cao sau chi phí.
- Drawdown trong ngưỡng chấp nhận.
- Kết quả ổn định qua nhiều giai đoạn.
- Robustness với spread/slippage.

Kết quả hiện tại chưa đạt các điều kiện đó. Vì vậy backtest chỉ nên nằm ở vai trò minh họa.

## 4.19. Bài học từ kết quả âm

Một điểm quan trọng của nghiên cứu là không phải mọi mô hình đề xuất đều thắng. Kết quả Hybrid Stacking không vượt LightGBM cho thấy:

1. Dữ liệu tài chính có tín hiệu yếu — phù hợp EMH [1].
2. Ensemble phức tạp có thể không giúp nếu base models không bổ sung lỗi cho nhau [18].
3. LightGBM là baseline rất mạnh cho feature tabular.
4. Label Hold thấp là nút thắt cần xử lý trước khi tăng complexity.
5. Báo cáo trung thực giúp tránh overclaim — phù hợp cảnh báo về backtest overfitting [7], [9].

Đây là kết quả có thể bảo vệ được nếu luận văn nhấn mạnh phương pháp luận thay vì lợi nhuận.

## 4.20. Đề xuất bảng/biểu đồ nên đưa vào bản Word

Khi chuyển sang báo cáo Word/PDF, nên có:

1. Sơ đồ pipeline 6 stage.
2. Bảng cấu hình label.
3. Bảng danh sách feature groups.
4. Bảng phân phối nhãn.
5. Bảng metrics tổng (§4.3).
6. Bảng per-class precision/recall/F1 (§4.3).
7. Confusion matrix (§4.5).
8. Bảng model comparison toàn diện (§4.4).
9. Biểu đồ walk-forward accuracy theo cửa sổ (§4.6).
10. Biểu đồ equity curve backtest (§4.9).
11. Bảng metric quality zones (§4.9).
12. Bảng hạn chế và hướng phát triển.

## 4.21. Kết luận thực nghiệm chi tiết

Thực nghiệm chứng minh pipeline hoạt động end-to-end và tạo được kết quả đo lường đầy đủ. Tuy nhiên mô hình chưa đạt hiệu quả đủ mạnh để triển khai giao dịch. Điểm mạnh của đồ án là quy trình đánh giá và báo cáo minh bạch; điểm yếu chính là chất lượng tín hiệu hướng và lớp Hold.

Tổng kết kết quả chính:

| Khía cạnh | Đánh giá |
|---|---|
| Pipeline end-to-end | ✅ Hoạt động đầy đủ 6 stage |
| Walk-forward validation | ✅ 25 cửa sổ, purge/embargo đúng chuẩn |
| Labeling triple-barrier | ✅ Nhưng Hold quá thấp (8.5%) |
| So sánh baseline | ✅ Đầy đủ 7 mô hình |
| Hiệu suất mô hình | ❌ Chưa vượt baseline (Dir. Acc. < 0.5) |
| Backtest demo | ⚠️ Dương nhẹ nhưng Sharpe thấp |
| Artifact đầy đủ | ✅ JSON/CSV/HTML có thể kiểm tra lại |
| Báo cáo trung thực | ✅ Không overclaim |

Kết quả này dẫn trực tiếp đến hướng phát triển: cải thiện label design, calibration, thêm dữ liệu vĩ mô và phân tích SHAP theo regime.
