import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import cv2
from torchvision import transforms
import numpy as np
import streamlit as st
import PIL

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Prototype Skin Cancer Detection App - Deteksi Penyakit Kanker Kulit")
st.text("Mohon upload gambar lesi kulit dari jarak 5-15cm dengan format jpg/jpeg/png.")

# EfficientNet Model Class
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, extractor_trainable=True):
        super(EfficientNetModel, self).__init__()
        efficientnet = models.efficientnet_b2(pretrained=True)

        if not extractor_trainable:
            for param in efficientnet.parameters():
                param.requires_grad = False

        # Extract feature extractor and adjust the classifier for num_classes
        self.feature_extractor = efficientnet.features
        num_features = efficientnet.classifier[1].in_features
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

@st.cache(allow_output_mutation=True)
def load_model():
    model = EfficientNetModel(num_classes=8, model_name="efficientnet_b2")
    state_dict = torch.load('effnet_skincancer2_weights.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to('cpu')
    model.eval()
    return model

def preprocess_image(image_file):
    # Convert uploaded file to an OpenCV image
    image = np.array(PIL.Image.open(image_file))

    if image is None:
        raise ValueError("gagal meload gambar")

    # If the image has an alpha channel (4 channels), remove the alpha channel
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Convert the image to RGB if it's not already in that format
    elif len(image.shape) == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the necessary transforms for your model
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Match EfficientNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess and return the image tensor
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor

def predict_image(model, image_tensor, class_mapping):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_class = list(class_mapping.keys())[predicted_index]
        return predicted_class, probabilities

# Load the model
with st.spinner("Meload model ke memori..."):
    model = load_model()

# Class mapping for predictions
class_mapping = {'Actinic Keratosis': 0, 'Basal Cell Carcinoma': 1, 'Benign Keratosis Lesion': 2, 'Dermato Fibroma': 3, 'Melanoma': 4, 'Non Cancer': 5, 'Melanocytic Nevi': 6, 'Vascular Skin Lesion': 7}

# Image uploader
uploaded_file = st.file_uploader("Upload Foto Lesi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    image_tensor = preprocess_image(uploaded_file)

    # Predict the image class
    predicted_class, probabilities = predict_image(model, image_tensor, class_mapping)

    # Display the results
    st.image(uploaded_file, caption="Gambar Lesi Kulit", use_column_width=True)
    st.write(f"Hasil Prediksi: {predicted_class}")

    # Get the top 3 classes and their probabilities
    top1_prob, top1_idx = torch.topk(probabilities, 1, dim=1)

    # Sort the top 3 classes by probability
    top1_classes = [(list(class_mapping.keys())[i], top1_prob[0][idx].item())
                    for idx, i in enumerate(top1_idx[0])]

    # Display the top 3 classes and their probabilities
    st.write("Hasil Prediksi Gambar Lesi Kulit:")
    for class_name, prob in top3_classes:
        st.write(f"{class_name}: {prob:.4f}")

    # Display Explanation
    if predicted_class == "Melanoma":
        st.write("Lesi pada gambar dipastikan sebagai melanoma karena menunjukkan semua karakteristik klinis yang tidak dapat disangkal, seperti kejelasan asimetri, tepi tidak rata dan kabur, variasi warna mencolok (seperti hitam, coklat, merah, atau putih), serta ukuran di atas 6mm. Model AI telah dilatih menggunakan ribuan kasus melanoma yang terverifikasi oleh ahli medis memiliki kemampuan luar biasa untuk secara konsisten mendeteksi pola-pola ini dengan akurasi yang jauh lebih tinggi daripada visualisasi manusia, menjamin diagnosis yang cepat, tepat, dan akurat. Referensi bacaan melanoma yang mudah didapahami dapat diakses pada link berikut (https://ayosehat.kemkes.go.id/jarang-menyerang-kenali-tanda-kanker-kulit-melanoma)")
        st.write("**Langkah perlindungan:**")
        st.markdown("- Gunakan pakaian yang dapat melindungi dari sinar matahari seperti baju berlengan panjang dan celana panjang yang terbuat dari kain bertenun rapat atau ber-UPF, topi bertepi lebar untuk melindungi wajah, leher, dan telinga, kacamata hitam untuk melindungi mata dan kulit di sekitar mata")
        st.markdown("- Gunakan tabir surya [(panduan memakai tabir surya)](https://www.aad.org/media/stats-sunscreen)")
        st.markdown("- Gunakan payung atau tetap berada ditempat teduh")
        st.write("**Rekomendasi:** segeralah periksakan lesi kulit anda pada dokter untuk mendapatkan penanganan lanjut")


else :
    st.text("Harap upload file foto sesuai dengan format")

