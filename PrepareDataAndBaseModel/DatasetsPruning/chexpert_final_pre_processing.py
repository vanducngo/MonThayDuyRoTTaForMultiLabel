import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =============================================================================
# CẤU HÌNH TRUNG TÂM
# =============================================================================
# Danh sách 5 bệnh mục tiêu chính
TARGET_DISEASES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pleural Effusion',
    'Pneumothorax',
]

# Các cột metadata không cần thiết sẽ bị xóa
METADATA_COLS_TO_DROP = ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']

# Tỷ lệ chia dữ liệu cho tập huấn luyện
TRAIN_RATIO = 0.8

# Random state để đảm bảo kết quả chia lặp lại được
RANDOM_STATE = 42


def step_1_clean_and_restructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 1: Dọn dẹp DataFrame ban đầu.
    - Đổi tên cột 'Path' thành 'image_id'.
    - Xóa các cột metadata.
    - Xử lý các giá trị không chắc chắn (-1.0) và giá trị thiếu (NaN) thành 0.
    """
    print("--- Bước 1: Bắt đầu dọn dẹp và tái cấu trúc dữ liệu ---")

    if 'Path' in df.columns:
        df = df.rename(columns={'Path': 'image_id'})
        print(" -> Đã đổi tên cột 'Path' thành 'image_id'.")
    
    cols_to_drop = [col for col in METADATA_COLS_TO_DROP if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f" -> Đã xóa các cột metadata: {cols_to_drop}")

    finding_cols = [col for col in df.columns if col != 'image_id']
    
    for col in finding_cols:
        df[col] = df[col].fillna(0.0)
        df[col] = df[col].replace(-1.0, 0.0)
    print(" -> Đã xử lý các nhãn không chắc chắn (-1.0) và NaN thành 0.0.")
    
    print("--- Bước 1: Hoàn tất ---\n")
    return df

def step_2_reduce_columns(df: pd.DataFrame, target_diseases: list) -> pd.DataFrame:
    """
    Bước 2: Giảm thiểu số cột, chỉ giữ lại 'image_id' và các bệnh mục tiêu.
    """
    print("--- Bước 2: Bắt đầu giảm thiểu số cột ---")
    
    columns_to_keep = ['image_id'] + target_diseases
    
    # Kiểm tra xem tất cả các cột cần giữ có tồn tại không
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Lỗi: Các cột mục tiêu sau không có trong dữ liệu: {missing_cols}")
        
    df_reduced = df[columns_to_keep]
    print(f" -> Đã giữ lại {len(df_reduced.columns)} cột: ['image_id'] và {len(target_diseases)} bệnh mục tiêu.")
    print("--- Bước 2: Hoàn tất ---\n")
    
    return df_reduced

def step_3_filter_positive_cases(df: pd.DataFrame, diseases: list) -> pd.DataFrame:
    """
    Bước 3: Lọc và chỉ giữ lại những dòng có ít nhất một bệnh dương tính.
    """
    print("--- Bước 3: Bắt đầu lọc các ca bệnh dương tính ---")
    
    initial_rows = len(df)
    # Lọc các dòng mà ít nhất một bệnh trong danh sách `diseases` có giá trị là 1.0
    df_filtered = df[df[diseases].eq(1.0).any(axis=1)].copy()
    
    final_rows = len(df_filtered)
    print(f" -> Đã lọc dữ liệu. Giữ lại {final_rows} / {initial_rows} dòng (loại bỏ các mẫu không có bệnh).")
    print("--- Bước 3: Hoàn tất ---\n")
    
    return df_filtered

def step_4_split_data(df: pd.DataFrame, target_diseases: list, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bước 4: Chia dữ liệu thành tập train và valid, lưu ra file và trả về các DataFrame.
    """
    print("--- Bước 4: Bắt đầu chia dữ liệu thành tập train và valid ---")
    
    # Sắp xếp lại cột để đảm bảo thứ tự nhất quán trước khi lưu
    # Reordering is implicitly done by step 2, but this ensures it
    df = df[['image_id'] + target_diseases]

    stratify_key = df[target_diseases].apply(lambda x: ''.join(x.astype(str)), axis=1)
    
    train_df, valid_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        stratify=stratify_key,
        random_state=RANDOM_STATE
    )
    
    train_output_path = os.path.join(output_dir, 'train_final.csv')
    valid_output_path = os.path.join(output_dir, 'valid_final.csv')
    
    train_df.to_csv(train_output_path, index=False)
    valid_df.to_csv(valid_output_path, index=False)
    
    print(f" -> Tập huấn luyện ({len(train_df)} mẫu) đã được lưu tại: {train_output_path}")
    print(f" -> Tập kiểm định ({len(valid_df)} mẫu) đã được lưu tại: {valid_output_path}")
    print("--- Bước 4: Hoàn tất ---\n")
    
    return train_df, valid_df

def step_5_visualize_distribution(train_df: pd.DataFrame, valid_df: pd.DataFrame, target_diseases: list, output_path: str):
    """
    Bước 5: Tạo và lưu biểu đồ so sánh phân bổ nhãn, hiển thị cả số lượng và phần trăm.
    """
    print("--- Bước 5: Bắt đầu tạo biểu đồ phân bổ nhãn chi tiết ---")
    
    # Tính toán số liệu
    train_counts = train_df[target_diseases].sum()
    valid_counts = valid_df[target_diseases].sum()
    train_total = len(train_df)
    valid_total = len(valid_df)
    train_percentages = (train_counts / train_total) * 100
    valid_percentages = (valid_counts / valid_total) * 100

    # Thiết lập cho biểu đồ
    x = np.arange(len(target_diseases))
    width = 0.35
    fig, ax = plt.subplots(figsize=(15, 9))
    
    # Vẽ các cột
    rects1 = ax.bar(x - width/2, train_counts, width, label=f'Train Set ({train_total} mẫu)')
    rects2 = ax.bar(x + width/2, valid_counts, width, label=f'Validation Set ({valid_total} mẫu)')
    
    # Thêm nhãn, tiêu đề
    ax.set_ylabel('Số lượng ca dương tính (Count)')
    ax.set_title('So sánh phân bổ nhãn bệnh (Số lượng và Tỷ lệ %)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(target_diseases, rotation=20, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    
    # Tạo nhãn tùy chỉnh (số lượng và phần trăm)
    train_labels = [f'{c}\n({p:.1f}%)' for c, p in zip(train_counts, train_percentages)]
    valid_labels = [f'{c}\n({p:.1f}%)' for c, p in zip(valid_counts, valid_percentages)]

    # Thêm nhãn trên đầu mỗi cột
    ax.bar_label(rects1, labels=train_labels, padding=5, fontsize=10)
    ax.bar_label(rects2, labels=valid_labels, padding=5, fontsize=10)

    # Điều chỉnh giới hạn trục Y để có thêm không gian cho nhãn
    ax.set_ylim(0, max(train_counts.max(), valid_counts.max()) * 1.2)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f" -> Biểu đồ chi tiết đã được lưu tại: {output_path}")
    print("--- Bước 5: Hoàn tất ---")


def main(input_csv_path: str):
    """
    Hàm chính điều phối toàn bộ quy trình xử lý dữ liệu.
    """
    print(f"Bắt đầu quy trình xử lý cho file: {input_csv_path}\n")
    
    if not os.path.exists(input_csv_path):
        print(f"LỖI: Không tìm thấy file đầu vào tại '{input_csv_path}'")
        return
        
    output_dir = os.path.dirname(input_csv_path)
    
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Đọc file thành công. Kích thước ban đầu: {df.shape}\n")
    except Exception as e:
        print(f"LỖI: Không thể đọc file CSV. Chi tiết: {e}")
        return

    # === THỰC HIỆN CÁC BƯỚC XỬ LÝ THEO LUỒNG MỚI ===
    
    df_step1 = step_1_clean_and_restructure(df)
    df_step2 = step_2_reduce_columns(df_step1, TARGET_DISEASES)
    df_step3 = step_3_filter_positive_cases(df_step2, TARGET_DISEASES)
    
    # Bước 4: Chia dữ liệu và lưu kết quả
    train_df, valid_df = step_4_split_data(df_step3, TARGET_DISEASES, output_dir)
    
    # Bước 5: Tạo biểu đồ
    chart_output_path = os.path.join(output_dir, 'label_distribution_final.png')
    step_5_visualize_distribution(train_df, valid_df, TARGET_DISEASES, chart_output_path)
    
    print("\n>>> QUY TRÌNH XỬ LÝ HOÀN TẤT! <<<")


if __name__ == "__main__":
    input_file = '/Users/admin/Working/Data/CheXpert-v1.0-small/train_origin.csv'    
    main(input_file)