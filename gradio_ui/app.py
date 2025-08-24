import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1)),
    ('bn1', nn.BatchNorm2d(32)),
    ('relu1', nn.ReLU(inplace=True)),
    ('mp1', nn.MaxPool2d(2)),

    ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
    ('bn2', nn.BatchNorm2d(64)),
    ('relu2', nn.ReLU(inplace=True)),
    ('mp2', nn.MaxPool2d(2)),

    ('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
    ('bn3', nn.BatchNorm2d(128)),
    ('relu3', nn.ReLU(inplace=True)),
    ('mp3', nn.MaxPool2d(2)),

    ('conv4', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
    ('bn4', nn.BatchNorm2d(256)),
    ('relu4', nn.ReLU(inplace=True)),
    ('mp4', nn.MaxPool2d(2)),

    ('flatten', nn.Flatten()),
    ('linear', nn.Linear(256*8*8, 512)), 
    ('reluf', nn.ReLU(inplace=True)),
    ('dropout', nn.Dropout(0.4)),
    ('linearf', nn.Linear(512, 10))
]))

model.load_state_dict(torch.load("gradio_ui/cnn_model (1).pth", map_location=device))

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.Grayscale(),        
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_sketch(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return f"Predicted class: {pred.item()}"

demo = gr.Interface(
    fn=predict_sketch,
    inputs=gr.Sketchpad(),  
    outputs="text",
    title="SketchDRW Classifier",
    description="Draw a sketch and see the predicted class."
)

if __name__ == "__main__":
    demo.launch(share=True)
