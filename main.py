import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import io
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Load pre-trained Bio_ClinicalBERT model and tokenizer
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # Pre-trained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=11)  # Set num_labels to match class_labels

import requests

API_URL = "https://router.huggingface.co/hf-inference/v1"
headers = {"Authorization": f"Bearer {API_KEY}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	


# Define class labels (replace with your actual disease labels)
class_labels = [
    "Flu", "Common Cold", "COVID-19", "Pneumonia", "Allergies",
    "Migraine", "Bronchitis", "Strep Throat", "Sinusitis", "Gastroenteritis", "Chest Infection"
]

def analyze_text(symptoms):
    # Tokenize input symptoms
    inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities

def get_top_predictions(probabilities, class_labels, top_k=3):
    num_classes = probabilities.shape[-1]  # Get the number of output classes
    top_k = min(top_k, num_classes)  # Ensure top_k does not exceed the number of classes
    
    # Get the top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=top_k)
    top_probs = top_probs.detach().numpy().flatten()
    top_indices = top_indices.detach().numpy().flatten()
    
    # Map indices to class labels
    top_labels = [class_labels[i] for i in top_indices]
    
    # Return as a list of tuples (label, probability)
    return list(zip(top_labels, top_probs))


def diagnose(symptoms):
    # Analyze symptoms
    text_predictions = analyze_text(symptoms)
    
    # Get top predictions
    top_predictions = get_top_predictions(text_predictions, class_labels)
    
    # Return results
    diagnosis = {
        "text_analysis": top_predictions
    }
    return diagnosis

# Streamlit app layout
st.title("AI-Powered Medical Diagnosis Assistant")

# Input for symptoms
symptoms = st.text_area("Enter patient symptoms:")

# Analyze button
if st.button("Analyze"):
    if symptoms:
        # Perform diagnosis
        diagnosis = diagnose(symptoms)

        # Display results
        st.subheader("Diagnosis:")
        
        st.write("**Text Analysis Results:**")
        for label, prob in diagnosis["text_analysis"]:
            st.write(f"- {label}: {prob * 100:.2f}%")
    else:
        st.error("Please enter symptoms.")
