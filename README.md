# Test-Time Adaptation for Dynamic and Multi-Label CXR Diagnosis
Thích ứng tại Thời điểm Kiểm thử cho Chẩn đoán X-quang Đa nhãn trong Kịch bản Động

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

Đây là repository chính thức cho dự án môn học **CS2225.CH190: Nhận dạng thị giác và ứng dụng**. Dự án này đề xuất **RoTTA-ML**, một sự tổng quát hóa của phương pháp **RoTTA (Robust Test-Time Adaptation)** cho bài toán phân loại đa nhãn (*multi-label classification*) và xác thực hiệu quả của nó trong lĩnh vực chẩn đoán hình ảnh y tế (CXR).

---

## 📖 Tổng quan (Overview)

Các mô hình AI chẩn đoán y tế thường bị suy giảm hiệu năng nghiêm trọng khi đối mặt với **distribution shift**—sự khác biệt giữa dữ liệu huấn luyện và dữ liệu thực tế. **Test-Time Adaptation (TTA)** là một giải pháp hứa hẹn cho phép mô hình tự thích ứng với dữ liệu mới mà không cần gán nhãn lại.

Tuy nhiên, hầu hết các phương pháp TTA hiện có, bao gồm cả SOTA như RoTTA, đều được thiết kế cho bài toán đơn nhãn. Nghiên cứu này đề xuất **RoTTA-ML**, một sự tổng quát hóa toàn diện của RoTTA, được tái thiết kế để hoạt động hiệu quả trong các kịch bản **vừa động (dynamic), vừa đa nhãn (multi-label)** của y tế thực tế.

### ✨ Đóng góp chính

-   **Framework `RoTTA-ML`:** Một hệ thống TTA hoàn chỉnh, có khả năng thích ứng liên tục cho các mô hình chẩn đoán đa nhãn.
-   **Memory Bank `MultiLabel-CSTU`:** Một kiến trúc *memory bank* mới với cấu trúc phẳng và thuật toán cân bằng thông minh, có khả năng duy trì sự đa dạng của từng nhãn bệnh và bảo vệ các lớp thiểu số.
-   **Mục tiêu Học nhất quán cho Đa nhãn:** Xây dựng lại cơ chế tạo *pseudo-label*, tính toán *uncertainty*, và hàm *consistency loss* (`bce_entropy`) dựa trên *Binary Cross-Entropy*.
-   **Benchmark Thực nghiệm:** Thiết lập một quy trình đánh giá nghiêm ngặt, nơi một mô hình được huấn luyện trên **CheXpert** phải thích ứng với luồng dữ liệu nhiễu động từ **NIH-14**, và so sánh với các baseline như **Source-only** và **TENT-ML**.

### 📊 Kết quả chính (Key Results)

Trong một kịch bản nhiễu động khắc nghiệt, `RoTTA-ML` đã chứng tỏ sự bền vững vượt trội:
| Phương pháp | Mean AUC | Thay đổi (Δ%) | Ghi chú |
| :--- | :---: | :---: | :--- |
| `Source-only` | 0.6389 | - | Baseline không thích ứng |
| `TENT-ML` | 0.5588 | **-12.54%** | **Sụp đổ hoàn toàn** |
| **`RoTTA-ML`** | **0.6615** | **+3.54%** | **Cải thiện đáng kể** |

---

## 👥 Nhóm phát triển

Dự án được thực hiện bởi các thành viên:
-   **Văn Đức Ngọ** - 240101020
-   **Phạm Thăng Long** - 240101016
-   **Nguyễn Hoàng Hải** - 240101008

**Giảng viên hướng dẫn:** PGS. Lê Đình Duy

---

## 🚀 Bắt đầu (Getting Started)

### 1. ⚙️ Cài đặt Môi trường

```bash
# 1. Clone repository
git clone https://github.com/vanducngo/CS2225.CH190.NhanDangThiGiacVaUngDung.git
cd CS2225.CH190.NhanDangThiGiacVaUngDung

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên macOS/Linux
# venv\Scripts\activate  # Trên Windows

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
