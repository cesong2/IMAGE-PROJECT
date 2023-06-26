from django.shortcuts import render
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image




def my_view(request):
    if request.method == 'POST':
        class_names = ['Atelectasis', 'Normal']
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
        ])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_file = request.FILES['image_file']  # 업로드된 이미지 파일을 가져옵니다.
        image = Image.open(image_file)
        image = image.convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)
        model = torch.load('C:/python/test/myproject2/test1/model/Atelectasis.h5')
        model.to(device)
        model.eval()
        # 예측 수행
        with torch.no_grad():
            output = model(input_image)

        # 예측 결과 확인
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_names[predicted_idx.item()]
        print(f"Predicted Label : {predicted_label}")



        #return render(request, 'c:/python/myproject/myapp/templates/my_template.html', {'prediction': prediction})
        return render(request, 'my_template.html', {'prediction': predicted_label})
    else:
        #return render(request, 'c:/python/myproject/myapp/templates/my_template.html')
        return render(request, 'my_template.html')