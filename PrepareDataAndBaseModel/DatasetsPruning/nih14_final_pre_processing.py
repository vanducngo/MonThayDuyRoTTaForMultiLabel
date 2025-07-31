import pandas as pd
import os
import shutil
from tqdm import tqdm
import glob
import argparse

# =============================================================================
# CẤU HÌNH TRUNG TÂM
# =============================================================================

# Định nghĩa các nhãn gốc trong NIH14 mà chúng ta muốn giữ lại
# Lưu ý: 'Effusion' trong NIH14 sẽ được ánh xạ thành 'Pleural Effusion'
DISEASES_TO_KEEP_NIH14 = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Effusion',  # Tên gốc trong NIH14
    'Pneumothorax'
]

# Định nghĩa bộ nhãn cuối cùng (chuẩn hóa) để đồng bộ với các bộ dữ liệu khác
FINAL_LABEL_SET = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pleural Effusion',
    'Pneumothorax'
]


def step_1_scan_image_paths(root_path: str) -> dict:
    """
    Bước 1: Quét tất cả các thư mục con 'images*' để lập chỉ mục đường dẫn ảnh.
    """
    print("--- Bước 1: Bắt đầu quét và lập chỉ mục các file ảnh ---")
    all_image_paths = {}
    # Tìm tất cả các thư mục có tên bắt đầu bằng 'images'
    image_folders = glob.glob(os.path.join(root_path, 'images*'))
    if not image_folders:
        raise FileNotFoundError(f"Không tìm thấy thư mục 'images*' nào trong: {root_path}")

    for folder in tqdm(image_folders, desc="Scanning image folders"):
        # Đường dẫn tới thư mục images thực sự bên trong (ví dụ: images_01/images)
        final_folder_path = os.path.join(folder, 'images')
        if not os.path.isdir(final_folder_path):
            print(f"Cảnh báo: Bỏ qua thư mục không hợp lệ: {final_folder_path}")
            continue
            
        for img_file in os.listdir(final_folder_path):
            if img_file.endswith('.png'):
                all_image_paths[img_file] = os.path.join(final_folder_path, img_file)
    
    print(f" -> Đã tìm thấy tổng cộng {len(all_image_paths)} ảnh.")
    print("--- Bước 1: Hoàn tất ---\n")
    return all_image_paths


def step_2_filter_and_standardize_csv(source_csv_path: str) -> pd.DataFrame:
    """
    Bước 2: Đọc file CSV, lọc các hàng chỉ chứa bệnh mục tiêu và chuẩn hóa nhãn.
    """
    print("--- Bước 2: Bắt đầu đọc, lọc và chuẩn hóa file CSV ---")
    
    if not os.path.exists(source_csv_path):
        raise FileNotFoundError(f"File CSV gốc không tồn tại: {source_csv_path}")

    df_raw = pd.read_csv(source_csv_path)
    df_raw.rename(columns={'Finding Labels': 'labels', 'Image Index': 'image_id'}, inplace=True)
    
    # Tạo điều kiện lọc: giữ lại một hàng nếu nhãn của nó chứa bất kỳ bệnh nào trong DISEASES_TO_KEEP_NIH14
    filter_condition = df_raw['labels'].str.contains('|'.join(DISEASES_TO_KEEP_NIH14))
    df_filtered = df_raw[filter_condition].copy()
    
    print(f" -> Số lượng bản ghi gốc: {len(df_raw)}")
    print(f" -> Số lượng bản ghi sau khi lọc: {len(df_filtered)}")

    # Tạo các cột nhãn nhị phân và chuẩn hóa tên
    print(" -> Đang tạo các cột nhãn nhị phân (one-hot encoded)...")
    for disease_final, disease_original in zip(FINAL_LABEL_SET, DISEASES_TO_KEEP_NIH14):
        df_filtered[disease_final] = df_filtered['labels'].apply(lambda x: 1 if disease_original in x else 0)

    # Chỉ giữ lại các cột cần thiết cho file CSV mới và sắp xếp lại
    columns_to_save = ['image_id'] + FINAL_LABEL_SET
    df_final = df_filtered[columns_to_save]

    print("--- Bước 2: Hoàn tất ---\n")
    return df_final


def step_3_copy_files_and_save_csv(df_final: pd.DataFrame, all_image_paths: dict, output_dir: str):
    """
    Bước 3: Sao chép các file ảnh đã lọc và lưu DataFrame cuối cùng.
    """
    print("--- Bước 3: Bắt đầu sao chép ảnh và lưu file CSV cuối cùng ---")
    
    target_image_dir = os.path.join(output_dir, 'images')
    target_csv_path = os.path.join(output_dir, 'validate.csv')
    os.makedirs(target_image_dir, exist_ok=True)
    
    num_copied = 0
    images_not_found = []
    
    for image_id in tqdm(df_final['image_id'], desc="Copying filtered images"):
        if image_id in all_image_paths:
            source_path = all_image_paths[image_id]
            target_path = os.path.join(target_image_dir, image_id)
            if not os.path.exists(target_path):
                shutil.copy2(source_path, target_path)
            num_copied += 1
        else:
            images_not_found.append(image_id)
    
    print(f" -> Hoàn tất sao chép. Đã sao chép: {num_copied} ảnh.")
    
    if images_not_found:
        print(f"Cảnh báo: {len(images_not_found)} ảnh không được tìm thấy và đã bị loại khỏi file CSV.")
        # Xóa các hàng tương ứng với ảnh không tìm thấy khỏi DataFrame
        df_final = df_final[~df_final['image_id'].isin(images_not_found)]
        
    print(f" -> Đang lưu file CSV mới vào: {target_csv_path}")
    df_final.to_csv(target_csv_path, index=False)
    
    print("--- Bước 3: Hoàn tất ---\n")


def main(nih_root_path: str, output_path: str):
    """
    Hàm chính điều phối toàn bộ quy trình xử lý dữ liệu NIH14.
    """
    print(f"Bắt đầu quy trình xử lý cho bộ dữ liệu NIH14")
    print(f"Thư mục nguồn: {nih_root_path}")
    print(f"Thư mục đích: {output_path}\n")

    try:
        # === THỰC HIỆN CÁC BƯỚC XỬ LÝ ===
        all_image_paths = step_1_scan_image_paths(nih_root_path)
        
        source_csv_path = os.path.join(nih_root_path, 'Data_Entry_2017.csv')
        df_processed = step_2_filter_and_standardize_csv(source_csv_path)
        
        step_3_copy_files_and_save_csv(df_processed, all_image_paths, output_path)
        
        print(">>> QUY TRÌNH XỬ LÝ NIH14 HOÀN TẤT! <<<")

    except Exception as e:
        print(f"\nĐÃ XẢY RA LỖI NGHIÊM TRỌNG: {e}")
        print("Quy trình đã bị dừng.")


if __name__ == "__main__":
    nih_path = '/Users/admin/Working/Data/CheXpert-v1.0-small/train_origin.csv'    
    output_path = '/Users/admin/Working/Data/CheXpert-v1.0-small/train_origin.csv'    
    main(nih_path, output_path)