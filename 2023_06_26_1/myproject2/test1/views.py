from django.shortcuts import render
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.utils import load_img
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
        ])
def Atelectasis(request):
    if request.method == 'POST':
        class_names = ['Atelectasis', 'Normal']
        image_file = request.FILES['image_file']  # 업로드된 이미지 파일을 가져옵니다.
        image = Image.open(image_file)
        image = image.convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)
        model = torch.load('test1/model/Atelectasis.h5')
        model.to(device)
        model.eval()
        # 예측 수행
        with torch.no_grad():
            output = model(input_image)
        # 예측 결과 확인
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_names[predicted_idx.item()]
        #return render(request, 'c:/python/myproject/myapp/templates/Atelectasis.html', {'prediction': prediction})
        return render(request, 'Atelectasis.html', {'prediction': predicted_label})
    else:
        #return render(request, 'c:/python/myproject/myapp/templates/Atelectasis.html')
        return render(request, 'Atelectasis.html')
def Cardiomegaly(request):
    if request.method == 'POST':
        # 모델 로드
        from tensorflow import keras
        print(1)
        model = keras.models.load_model('test1/model/Cardiomegaly_model.h5', compile=False)
        print(2)
        # 모델 컴파일
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(3)
        class_names = ['Normal', 'Cardiomegaly']
        img_size = (299, 299, 3)
        print(4)
        image_file = str(request.FILES['image_file'])
        print(image_file)
        img = load_img(image_file, target_size=img_size)
        print(img)
        img = np.array(img)
        img = np.reshape(img, (1, 299, 299, 3))

        predicted_label = class_names[np.argmax(model.predict(img))]
        return render(request, 'Cardiomegaly.html', {'prediction': predicted_label})
    else:
        return render(request, 'Cardiomegaly.html')
def Edema(request):
    if request.method == 'POST':
        class_names = ['Edema', 'Normal']
        # 모델 정의 및 불러오기
        model = torch.load('test1/model/Edema.h5')
        model.to(device)
        model.eval()
        # 이미지 파일 불러오기
        image_file = request.FILES['image_file']
        image = Image.open(image_file)  # 이미지 파일 경로 설정
        image = image.convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)
        # 예측 수행
        with torch.no_grad():
            output = model(input_image)
        # 예측 결과 확인
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_names[predicted_idx.item()]
        return render(request, 'Edema.html', {'prediction': predicted_label})
    else:
        return render(request, 'Edema.html')
def Effusion2(request):
    if request.method == 'POST':
        class_names = ['Effusion', 'Normal']
        # 모델 정의 및 불러오기
        model = torch.load('test1/model/Effusion.h5')
        model.to(device)
        model.eval()
        # 이미지 파일 불러오기
        image_file = request.FILES['image_file']
        image = Image.open(image_file)  # 이미지 파일 경로 설정
        image = image.convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)
        # 예측 수행
        with torch.no_grad():
            output = model(input_image)
        # 예측 결과 확인
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_names[predicted_idx.item()]
        return render(request, 'Effusion2.html', {'prediction': predicted_label})
    else:
        return render(request, 'Effusion2.html')
def Fibrosis(request):
    if request.method == 'POST':
        class_names = ['Fibrosis', 'Normal']
        # 모델 정의 및 불러오기
        model = torch.load('test1/model/Fibrosis.h5')
        model.to(device)
        model.eval()
        # 이미지 파일 불러오기
        image_file = request.FILES['image_file']
        image = Image.open(image_file)  # 이미지 파일 경로 설정
        image = image.convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)
        # 예측 수행
        with torch.no_grad():
            output = model(input_image)
        # 예측 결과 확인
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_names[predicted_idx.item()]
        return render(request, 'Fibrosis.html', {'prediction': predicted_label})
    else:
        return render(request, 'Fibrosis.html')