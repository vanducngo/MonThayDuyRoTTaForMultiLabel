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

## ⚙️ Cài đặt

### 1. Clone Repository

```bash
git clone git@github.com:vanducngo/CS2225.CH190.NhanDangThiGiacVaUngDung.git
cd CS2225.CH190.NhanDangThiGiacVaUngDung
```

### 2. Tạo Môi trường Ảo và Cài đặt Thư viện

Nên sử dụng một môi trường ảo để tránh xung đột thư viện.

```bash
# Tạo môi trường ảo (ví dụ với venv)
python -m venv venv

# Kích hoạt môi trường
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## 📂 Chuẩn bị Dữ liệu

### 1. Tải Dữ liệu

Tải hai bộ dữ liệu từ Kaggle và giải nén vào một thư mục (ví dụ: `datasets/`):

- **CheXpert (Miền nguồn):** [https://www.kaggle.com/datasets/ashery/chexpert](https://www.kaggle.com/datasets/ashery/chexpert)
- **NIH Chest X-ray14 (Miền đích):** [https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)

Cấu trúc thư mục của bạn nên trông như sau:
```
CS2225.CH190.NhanDangThiGiacVaUngDung/
├── datasets/
│   ├── CheXpert-v1.0-small/
│   └── nih-chest-xrays/
├── DataPreprocessing/
├── RoTTA/
└── ...
```

### 2. Chạy Script Tiền xử lý

Các script này sẽ lọc 5 bệnh lý mục tiêu, xử lý hình ảnh và tạo ra các file `.csv` cần thiết cho quá trình huấn luyện và kiểm thử.

```bash
cd DataPreprocessing

# Tiền xử lý CheXpert
python chexpert_final_pre_processing_train_validation_test.py

# Tiền xử lý NIH-14
python nih14_final_pre_processing.py
```
Sau khi chạy, các file dữ liệu đã được xử lý sẽ được tạo ra, sẵn sàng cho các bước tiếp theo.

## 🚀 Chạy Thí nghiệm (Running Experiments)

### 1. Cấu hình Đường dẫn Dữ liệu

Trước khi chạy, bạn **bắt buộc** phải cập nhật đường dẫn tới dữ liệu đã được tiền xử lý trong các file cấu hình YAML. Mở các file sau và chỉnh sửa các trường `DATA_DIR`:

- `RoTTA/configs/adapter/zero_shot.yaml`
- `RoTTA/configs/adapter/rotta.yaml`
- `RoTTA/configs/adapter/tent.yaml`

Ví dụ trong `rotta.yaml`:
```yaml
DATASET:
  NAME: "nih14" 
  # Sửa đường dẫn này thành đường dẫn tới thư mục chứa file csv và ảnh của NIH-14 đã được xử lý
  DATA_DIR: "/path/to/your/processed/nih14_data" 
  ...
```

### 2. Thực thi các Phương pháp

Di chuyển vào thư mục `RoTTA` và chạy các lệnh sau để thực thi từng phương pháp:

```bash
cd RoTTA
```

- **Chạy Baseline `Source-only` (Zero-Shot):**
    ```bash
    python ptta_multilabels.py -cfg configs/adapter/zero_shot.yaml
    ```

- **Chạy `RoTTA-ML` (Phương pháp đề xuất):**
    ```bash
    python ptta_multilabels.py -cfg configs/adapter/rotta.yaml
    ```

- **Chạy `TENT-ML` (Phương pháp so sánh):**
    ```bash
    python ptta_multilabels.py -cfg configs/adapter/tent.yaml
    ```

Kết quả và các file log sẽ được lưu trong thư mục `output/` được định nghĩa trong file config.

## 📄 Giấy phép (License)

Dự án này được cấp phép theo [MIT License](LICENSE).
