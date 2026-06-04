import pandas as pd
import os
import shutil
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def prepare_dataset(base_dir):
    csv_path = os.path.join(base_dir, 'HAM10000_metadata.csv')
    df = pd.read_csv(csv_path)

    # 1. Điền giá trị thiếu (Handling missing values)
    df['age'] = df['age'].fillna(df['age'].median())
    df['sex'] = df['sex'].fillna('unknown')
    df['localization'] = df['localization'].fillna('unknown')
    df['dx_type'] = df['dx_type'].fillna('unknown')

    # Ánh xạ từ vựng sang tiếng Việt
    localization_map = {
        'back': 'lưng', 'lower extremity': 'chi dưới', 'trunk': 'thân mình',
        'upper extremity': 'chi trên', 'abdomen': 'bụng', 'face': 'mặt',
        'chest': 'ngực', 'foot': 'bàn chân', 'unknown': 'vị trí không xác định',
        'neck': 'cổ', 'scalp': 'da đầu', 'hand': 'bàn tay', 'ear': 'tai',
        'genital': 'vùng sinh dục', 'acral': 'đầu chi'
    }

    # Ánh xạ phương pháp chẩn đoán
    dx_type_map = {
        'histo': 'mô bệnh học (sinh thiết)',
        'consensus': 'hội chẩn chuyên gia',
        'confocal': 'kính hiển vi đồng tiêu',
        'follow_up': 'theo dõi lâm sàng',
        'unknown': 'phương pháp không xác định'
    }
    
    # 2. Ánh xạ nhãn cố định
    label_map = {
        'akiec': 0,
        'bcc': 1,
        'bkl': 2,
        'df': 3,
        'mel': 4,
        'nv': 5,
        'vasc': 6
    }
    
    # Ghi label map ra file JSON
    label_map_path = os.path.join(base_dir, 'label_map.json')
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
    print(f"Saved label map to: {label_map_path}")
    
    # 3. Tạo câu mô tả chi tiết tiếng Việt (Multimodal Text)
    text_descriptions = []
    labels = []
    for index, row in df.iterrows():
        age = row['age']
        
        # Sex mapping
        if row['sex'] == 'male':
            sex = 'nam'
        elif row['sex'] == 'female':
            sex = 'nữ'
        else:
            sex = 'không rõ giới tính'
            
        loc = localization_map.get(row['localization'], 'vị trí không xác định')
        dx_type_text = dx_type_map.get(row['dx_type'], 'phương pháp không xác định')
        
        # Sinh câu văn
        text = f"Bệnh nhân {sex}, {int(age)} tuổi, xuất hiện tổn thương da ở vùng {loc}. Tổn thương được chẩn đoán thông qua {dx_type_text}."
            
        text_descriptions.append(text)
        labels.append(label_map[row['dx']])

    df['text_description'] = text_descriptions
    df['label'] = labels

    # 4. Chia tập dữ liệu (Train: 80%, Val: 10%, Test: 10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    # Lưu file CSV mới
    train_csv = os.path.join(base_dir, 'train_dataset.csv')
    val_csv = os.path.join(base_dir, 'val_dataset.csv')
    test_csv = os.path.join(base_dir, 'test_dataset.csv')
    
    train_df.to_csv(train_csv, index=False, encoding='utf-8')
    val_df.to_csv(val_csv, index=False, encoding='utf-8')
    test_df.to_csv(test_csv, index=False, encoding='utf-8')
    
    print(f"Created Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)}) CSV splits successfully.")

    # Gom ảnh từ 2 thư mục vào 1 thư mục chung 'all_images' để dễ load
    img_dir_1 = os.path.join(base_dir, 'HAM10000_images_part_1')
    img_dir_2 = os.path.join(base_dir, 'HAM10000_images_part_2')
    out_img_dir = os.path.join(base_dir, 'all_images')
    
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
        print("Copying images into a single directory...")
        for d in [img_dir_1, img_dir_2]:
            if os.path.exists(d):
                for img_name in tqdm(os.listdir(d), desc=f"Copying {os.path.basename(d)}"):
                    src = os.path.join(d, img_name)
                    dst = os.path.join(out_img_dir, img_name)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
        print("Done copying images!")
    else:
        print("The all_images directory already exists.")

if __name__ == '__main__':
    # Đường dẫn tới thư mục dataset
    base_dir = r'd:\TRI_TUE_NHAN_TAO\DU_AN_BTL\dataset'
    prepare_dataset(base_dir)
