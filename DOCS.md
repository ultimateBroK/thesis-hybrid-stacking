# Ghi chú đồng bộ tài liệu

Tài liệu đã được cập nhật theo runtime hiện tại của repo.

## Quyết định kiến trúc hiện tại

Không dùng Hybrid GRU + LightGBM làm runtime chính nữa. Runtime hiện tại là:

```text
Classic Hybrid Stacking
= Logistic Regression + Random Forest + LightGBM
  -> meta-model Logistic Regression
  -> Short / Hold / Long
```

## Cách viết trong luận văn

Nên mô tả đề tài là pipeline học máy dự báo tín hiệu XAU/USD H1 với đánh giá time-series có kiểm soát:

- feature causal;
- triple-barrier labeling;
- walk-forward validation;
- purge/embargo chống leakage;
- so sánh baseline và mô hình;
- feature importance/SHAP hoặc importance-based explanation.

Backtest chỉ là minh họa ứng dụng, không phải bằng chứng chính.

## Kết quả gần nhất

```text
Session: results/XAUUSD_1H_20260511_231114/
Hybrid Stacking accuracy: 0.3397
Hybrid Stacking macro F1: 0.3162
LightGBM accuracy: 0.3770
LightGBM macro F1: 0.3281
Backtest demo win rate: 47.67%
```

Diễn giải trung thực: Hybrid Stacking không luôn vượt LightGBM trên dữ liệu tài chính nhiễu cao. Đóng góp chính là quy trình đánh giá có kiểm soát và minh bạch.
