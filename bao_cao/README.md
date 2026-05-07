# BÁO CÁO ĐỒ ÁN

## Dự báo xu hướng ngắn hạn XAU/USD bằng mô hình Hybrid GRU + LightGBM với Triple Barrier Labeling

---

### Bố cục

| Chương | File | Nội dung | Số trang dự kiến |
|---|---|---|---|
| Chương 1 | `chuong1_gioi_thieu.md` | Giới thiệu | 8–10 |
| Chương 2 | `chuong2_co_so_ly_thuyet.md` | Cơ sở lý thuyết | 15–18 |
| Chương 3 | `chuong3_du_lieu.md` | Dữ liệu và tiền xử lý | 10–12 |
| Chương 4 | `chuong4_phuong_phap.md` | Phương pháp đề xuất | 14–16 |
| Chương 5 | `chuong5_thuc_nghiem.md` | Thực nghiệm và đánh giá | 12–15 |
| Chương 6 | `chuong6_minh_hoa.md` | Minh họa ứng dụng | 8–10 |
| Chương 7 | `chuong7_ket_luan.md` | Kết luận | 5–6 |
| Phụ lục | `tai_lieu_tham_khao.md` | Tài liệu tham khảo (57 nguồn) | 5–6 |

**Tổng số trang dự kiến: 77–93 trang**

---

### Hướng dẫn sử dụng

1. **Đọc theo thứ tự:** Chương 1 → 2 → 3 → 4 → 5 → 6 → 7
2. **Chương 5** cần điền kết quả thực tế sau khi chạy `pixi run workflow --force`
3. **Chương 6** cần điền kết quả backtest sau khi chạy pipeline
4. **Tài liệu tham khảo** ở cuối mỗi chương tương ứng với số thứ tự trong `tai_lieu_tham_khao.md`

### Format chuyển đổi

Các file Markdown có thể chuyển sang PDF/DOCX bằng:

```bash
# Pandoc (recommended)
pandoc chuong*.md -o bao_cao.pdf --pdf-engine=xelatex -V mainfont="Times New Roman" -V fontsize=12pt

# Hoặc từng chương riêng
pandoc chuong1_gioi_thieu.md -o chuong1.pdf
```

### Ghi chú

- Mỗi chương chứa inline references (trong ngoặc vuông) tương ứng với danh sách ở cuối chương.
- Danh sách tham khảo đầy đủ (57 nguồn, 11 nhóm) nằm trong `tai_lieu_tham_khao.md`; số thứ tự trong file đó khớp với số trong ngoặc vuông ở từng chương.
- Các bảng kết quả (Chapter 5, 6) có placeholder `—` cần điền sau khi chạy pipeline.
