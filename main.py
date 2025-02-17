import streamlit as st
from PIL import Image
from transformers import pipeline
from torchvision import models, transforms
import torch

# Load LLM for text analysis
text_analyzer = pipeline("text-generation", model="gpt-4")  # Replace with your LLM

# Load ResNet for image analysis
model = models.resnet50(pretrained=True)
model.eval()

# Preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Analyze symptoms
def analyze_symptoms(symptoms):
    prompt = f"Based on the following symptoms, suggest possible diagnoses: {symptoms}"
    response = text_analyzer(prompt, max_length=200)
    return response[0]['generated_text']

# Analyze image
def analyze_image(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
    return output

# Multimodal diagnosis
def multimodal_diagnosis(symptoms, image_path):
    text_diagnosis = analyze_symptoms(symptoms)
    image_analysis = analyze_image(image_path)
    combined_diagnosis = f"Text Analysis: {text_diagnosis}\nImage Analysis: {image_analysis}"
    return combined_diagnosis

# Streamlit app
st.title("AI-Powered Medical Diagnosis Assistant")

# Input for symptoms
symptoms = st.text_input("Enter patient symptoms:")

# Input for medical image
uploaded_file = st.file_uploader("Upload a medical image (X-ray, MRI):", type=["jpg", "png", "jpeg"])

# Analyze button
if st.button("Analyze"):
    if symptoms and uploaded_file:
        # Save the uploaded image
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Perform multimodal diagnosis
        diagnosis = multimodal_diagnosis(symptoms, image_path)
        
        # Display results
        st.subheader("Diagnosis:")
        st.write(diagnosis)
        
        # Display the uploaded image
        st.image(image_path, caption="Uploaded Image", use_column_width=True)
    else:
        st.error("Please enter symptoms and upload an image.")
