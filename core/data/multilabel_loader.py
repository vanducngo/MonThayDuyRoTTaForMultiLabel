import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

from ..utils.metrics import AUCProcessor

class CleanSingleDomainDataset(Dataset):
    """
    Lớp Dataset chỉ để đọc MỘT domain dữ liệu "sạch" duy nhất.
    Việc áp dụng nhiễu sẽ được thực hiện bên ngoài.
    """
    def __init__(self, cfg, transform=None):
        base_domain_cfg = cfg.DATASET.BASE_DOMAIN
        self.root_dir = os.path.join(base_domain_cfg.PATH, base_domain_cfg.IMAGE_DIR)
        self.transform = transform
        self.labels_list = cfg.DATASET.LABELS_LIST
        
        csv_path = os.path.join(base_domain_cfg.PATH, base_domain_cfg.CSV)
        print(f"Loading clean data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        self.image_col = 'image_id'
            
        print(f"Initialized clean dataset with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.image_col])
        
        # Xử lý các trường hợp đường dẫn khác nhau
        if ("CheXpert-v1.0-small" in img_name):
            img_path = f"/home/ngoto/Working/Data/{img_name}"
        elif not os.path.isabs(img_name):
             img_path = os.path.join(self.root_dir, img_name)
        else: # Nếu đường dẫn đã là tuyệt đối
             img_path = img_name
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found at {img_path}. Returning None.")
            return None
            
        labels = torch.tensor(row[self.labels_list].values.astype('float'), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        # Trả về tên domain gốc để tham khảo nếu cần, không bắt buộc
        return {'image': image, 'label': labels, 'domain': 'clean_base'}

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {'image': torch.empty(0), 'label': torch.empty(0), 'domain': []}
    return torch.utils.data.dataloader.default_collate(batch)


def build_loader_multilabel(cfg):
    """
    Xây dựng DataLoader chỉ để nạp dữ liệu SẠCH.
    Logic nhiễu và domain sẽ được xử lý trong vòng lặp chính.
    """
    # Chỉ cần định nghĩa transform cơ bản để chuẩn bị dữ liệu
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    clean_dataset = CleanSingleDomainDataset(cfg, transform=base_transform)
    
    loader = DataLoader(
        clean_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,  # Quan trọng: KHÔNG shuffle để duyệt qua dataset theo thứ tự
        num_workers=cfg.LOADER.NUM_WORKS,
        collate_fn=collate_fn_skip_none,
        pin_memory=True
    )
    
    # Processor vẫn được tạo ở đây như cũ
    # LABELS_LIST phải chứa tất cả các nhãn mà mô hình được huấn luyện
    result_processor = AUCProcessor(num_classes=len(cfg.DATASET.LABELS_LIST))
    
    print("Clean data loader and AUC processor are built successfully.")
    return loader, result_processor