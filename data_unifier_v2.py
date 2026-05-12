import logging
import os
import yaml

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED_DIR = os.path.join(WORKSPACE_DIR, 'unified_dataset')
YOLO_DIR = os.path.join(WORKSPACE_DIR, 'datasets/Fruits-detection')
SOURCES = [
    os.path.join(WORKSPACE_DIR, 'archive/fruits-360_100x100/fruits-360/Training'),
    os.path.join(WORKSPACE_DIR, 'datasets/images'),
    os.path.join(WORKSPACE_DIR, 'datasets/fruit-dataset'),
    os.path.join(WORKSPACE_DIR, 'datasets/archive')
]

MAPPING = {
    'Apple': 'Apple Red 1',
    'Banana': 'Banana 1',
    'Grape': 'Grape White 1',
    'Orange': 'Orange 1',
    'Pineapple': 'Pineapple 1',
    'Watermelon': 'Watermelon 1'
}

logging.basicConfig(level=logging.INFO, format='%(message)s')


def unify_all():
    os.makedirs(UNIFIED_DIR, exist_ok=True)
    total = 0

    for source in SOURCES:
        if not os.path.exists(source):
            continue
        logging.info(f"Taranıyor: {os.path.basename(source)}")
        for root, dirs, files in os.walk(source):
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                continue

            clean_name = os.path.basename(root).replace(' fruit', '').capitalize()
            if 'fruits-360' in root or 'Training' in root:
                clean_name = os.path.basename(root)

            if clean_name.lower() in ['train', 'test', 'valid', 'images', 'archive']:
                parts = root.split(os.sep)
                if len(parts) > 1:
                    clean_name = parts[-2].capitalize()

            if clean_name.lower() in ['train', 'test', 'valid', 'images']:
                continue

            target_path = os.path.join(UNIFIED_DIR, clean_name)
            os.makedirs(target_path, exist_ok=True)

            for img in images:
                src_img = os.path.join(root, img)
                unique_name = f"{os.path.basename(source)}_{clean_name}_{img}"
                dest_img = os.path.join(target_path, unique_name)
                try:
                    if not os.path.exists(dest_img):
                        os.symlink(src_img, dest_img)
                        total += 1
                except Exception:
                    pass

    if os.path.exists(YOLO_DIR):
        logging.info("Fruits-detection verileri işleniyor...")
        yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                class_names = data['names']

            for phase in ['train', 'valid', 'test']:
                img_dir = os.path.join(YOLO_DIR, phase, 'images')
                lbl_dir = os.path.join(YOLO_DIR, phase, 'labels')
                if not os.path.exists(img_dir):
                    continue

                for img_file in os.listdir(img_dir):
                    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    lbl_file = os.path.splitext(img_file)[0] + '.txt'
                    lbl_path = os.path.join(lbl_dir, lbl_file)

                    if os.path.exists(lbl_path):
                        with open(lbl_path, 'r') as f:
                            lines = f.readlines()
                            if not lines:
                                continue
                            class_id = int(lines[0].split()[0])
                            actual_class = class_names[class_id]
                            target_category = MAPPING.get(actual_class, actual_class)

                            target_path = os.path.join(UNIFIED_DIR, target_category)
                            os.makedirs(target_path, exist_ok=True)

                            src_img = os.path.join(img_dir, img_file)
                            dest_img = os.path.join(target_path, f"yolo_{phase}_{img_file}")

                            try:
                                if not os.path.exists(dest_img):
                                    os.symlink(src_img, dest_img)
                                    total += 1
                            except Exception:
                                pass

    logging.info(f"Toplam {total} görsel birleştirildi.")


if __name__ == '__main__':
    unify_all()
