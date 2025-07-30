import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# 🔁 ต้องเหมือนตอนเทรน
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 📂 โหลดรูป
image = Image.open("img_01.jpg").convert("RGB")
image = transform(image)
image = image.unsqueeze(0)  # เพิ่ม batch dimension

# 🧠 โมเดลต้องเหมือนที่เทรน
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64), nn.ReLU(),
            nn.Linear(64, 2)  # 2 class
        )
    def forward(self, x):
        return self.net(x)

# 🧠 โหลดโมเดล
model = SimpleCNN()
model.load_state_dict(torch.load("fruit_model.pth", map_location=torch.device("cpu")))
model.eval()

# 🔍 ทำนาย
with torch.no_grad():
    output = model(image)
    predicted_class = output.argmax(dim=1).item()

# 🏷️ แปลง index -> ชื่อผลไม้
class_names = ['apple', 'orange']
print(f"Prediction: {class_names[predicted_class]}")
