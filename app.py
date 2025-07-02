import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import hashlib
import os

# モデル定義
class PikachuResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    model = PikachuResNet50(num_classes=2)
    model.load_state_dict(torch.load("pikachu_classifier_resnet50.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@st.cache_data
def load_known_hashes():
    if os.path.exists("known_train_images.txt"):
        with open("known_train_images.txt") as f:
            return set(line.strip() for line in f)
    return set()

known_hashes = load_known_hashes()

def hash_image(image):
    return hashlib.md5(image.tobytes()).hexdigest()

# UI
st.title("⚡ 学習されたピカチュウを当てるなゲーム 🔥")
threshold = st.slider("アウト判定のしきい値（確信度 %）", 90, 100, 95) / 100
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_hash = hash_image(image)

    if img_hash in known_hashes:
        st.error("❌ アウト！これは学習に使われた画像です！（画像は表示されません）")
    else:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0][pred].item()

        if pred == 1 and confidence > threshold:
            st.error(f"❌ アウト！ピカチュウと強く予測（確信度: {confidence*100:.1f}%）")
            # 表示しない（意図的に画像非表示）
        else:
            st.image(image, caption="アップロード画像", use_container_width=True)
            st.success("✅ セーフ！これは学習されていない画像です")
            st.info(f"モデル予測: {'ピカチュウ' if pred == 1 else 'その他'}（確信度: {confidence*100:.1f}%）")
