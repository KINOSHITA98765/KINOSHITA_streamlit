import os
from PIL import Image
import hashlib

# 🔧 画像からハッシュを生成
def hash_image(image_path):
    with Image.open(image_path).convert("RGB") as img:
        return hashlib.md5(img.tobytes()).hexdigest()

# 📁 学習画像フォルダのパス（必要に応じて変更）
train_dir = "data/train"

hashes = []

# 📂 クラスごとのサブフォルダを処理
for cls in os.listdir(train_dir):
    cls_path = os.path.join(train_dir, cls)
    
    # 📛 フォルダでないもの（例: .DS_Store）をスキップ
    if not os.path.isdir(cls_path):
        continue

    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)

        # 📛 画像ファイル以外を無視
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            h = hash_image(img_path)
            hashes.append(h)
        except Exception as e:
            print(f"⚠️ エラー: {img_path} - {e}")

# 📄 ハッシュをテキストファイルに保存
with open("known_train_images.txt", "w") as f:
    for h in hashes:
        f.write(h + "\n")

print("✅ known_train_images.txt を作成しました！")
