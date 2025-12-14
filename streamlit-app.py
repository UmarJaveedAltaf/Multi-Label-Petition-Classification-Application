import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
import os
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Petition Classification Tool",
    page_icon="📝",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .top-category {
        text-align: center;
        padding: 20px;
        margin: 15px 0;
        background-color: #e3f2fd;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .category-name {
        color: #1E88E5;
        margin-bottom: 5px;
        font-size: 2rem;
        font-weight: bold;
    }
    .confidence-score {
        font-size: 1.2rem;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<div class='main-header'>Petition Classification Tool</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Analyze and classify petition text using NLP models</div>", unsafe_allow_html=True)

# EUROVOC Category mapping with links
def get_eurovoc_link(category_id):
    """Get the EUROVOC link for a category ID"""
    return f"https://op.europa.eu/en/web/eu-vocabularies/concept/-/resource?uri=http://eurovoc.europa.eu/{category_id}"

# Model classes (required for loading saved models)
class BERTGRUClassifier(nn.Module):
    def __init__(self, output_dim, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = output.last_hidden_state
        x, _ = self.gru(x)
        pooled = torch.mean(x, dim=1)
        x = self.dropout(pooled)
        return torch.sigmoid(self.fc(x))

class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, output_dim, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bilstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = output.last_hidden_state
        x, _ = self.bilstm(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load models and vectorizers
@st.cache_resource
def load_models():
    models = {}
    
    # Load TF-IDF vectorizer
    tfidf_path = "tfidf_vectorizer.pkl"
    if os.path.exists(tfidf_path):
        models["tfidf_vectorizer"] = joblib.load(tfidf_path)
    else:
        models["tfidf_vectorizer"] = TfidfVectorizer(max_features=10000)
    
    # Load MultiLabelBinarizer
    mlb_path = "multilabel_binarizer.pkl"
    if os.path.exists(mlb_path):
        models["multilabel_binarizer"] = joblib.load(mlb_path)
    else:
        models["multilabel_binarizer"] = MultiLabelBinarizer()
    
    # Load Naive Bayes model
    nb_path = "naive_bayes_model.pkl"
    if os.path.exists(nb_path):
        models["Naive Bayes"] = joblib.load(nb_path)
    else:
        models["Naive Bayes"] = OneVsRestClassifier(MultinomialNB())
    
    # Load Passive Aggressive Classifier
    pa_path = "passive_aggressive_model.pkl"
    if os.path.exists(pa_path):
        models["Passive Aggressive"] = joblib.load(pa_path)
    else:
        models["Passive Aggressive"] = OneVsRestClassifier(PassiveAggressiveClassifier())
    
    # Define a list of possible BERT model filenames to check
    bert_gru_paths = ["bert_gru_model.pt", "model_bert_gru.pt", "bert_gru.pt"]
    bert_gru_loaded = False
    
    # Get num_classes from the multilabel_binarizer or use default
    num_classes = 100
    if hasattr(models["multilabel_binarizer"], "classes_"):
        num_classes = len(models["multilabel_binarizer"].classes_)
    
    # Load BERT+GRU model
    for path in bert_gru_paths:
        if os.path.exists(path):
            try:
                models["BERT+GRU"] = torch.load(path, map_location=device, weights_only=False)
                bert_gru_loaded = True
                break
            except:
                try:
                    torch.serialization.add_safe_globals([BERTGRUClassifier])
                    models["BERT+GRU"] = torch.load(path, map_location=device)
                    bert_gru_loaded = True
                    break
                except:
                    pass
    
    if not bert_gru_loaded:
        models["BERT+GRU"] = BERTGRUClassifier(output_dim=num_classes).to(device)
    
    # Load BERT+BiLSTM model
    bert_bilstm_paths = ["bert_bilstm_model.pt", "model_bilstm.pt", "bert_bilstm.pt"]
    bert_bilstm_loaded = False
    
    for path in bert_bilstm_paths:
        if os.path.exists(path):
            try:
                models["BERT+BiLSTM"] = torch.load(path, map_location=device, weights_only=False)
                bert_bilstm_loaded = True
                break
            except:
                try:
                    torch.serialization.add_safe_globals([BERTBiLSTMClassifier])
                    models["BERT+BiLSTM"] = torch.load(path, map_location=device)
                    bert_bilstm_loaded = True
                    break
                except:
                    pass
    
    if not bert_bilstm_loaded:
        models["BERT+BiLSTM"] = BERTBiLSTMClassifier(output_dim=num_classes).to(device)
    
    # Load tokenizer for BERT-based models
    models["tokenizer"] = BertTokenizer.from_pretrained("bert-base-uncased")
    
    return models

# Load models
with st.spinner('Loading models...'):
    models = load_models()

# Sidebar
st.sidebar.markdown("## Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a classification model",
    ["Naive Bayes", "Passive Aggressive", "BERT+GRU", "BERT+BiLSTM"]
)

# Main area
st.markdown("### Enter Petition Text")
petition_text = st.text_area(
    "Paste the petition text here:",
    height=200
)

# Process button
if st.button("Classify Petition"):
    if not petition_text:
        st.error("Please enter petition text to classify.")
    else:
        with st.spinner("Analyzing petition..."):
            # Preprocess the text
            preprocessed_text = preprocess_text(petition_text)
            
            # Make predictions
            if selected_model in ["Naive Bayes", "Passive Aggressive"]:
                # Transform text using TF-IDF
                X = models["tfidf_vectorizer"].transform([preprocessed_text])
                
                # Predict
                predictions = models[selected_model].predict(X)
                
                # Get class indices
                if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
                    predicted_indices = np.where(predictions[0] == 1)[0]
                else:
                    predicted_indices = [predictions[0]]
                
                # Get confidence scores
                try:
                    sorted_classes = []
                    
                    if hasattr(models[selected_model], "predict_proba"):
                        confidence_scores = models[selected_model].predict_proba(X)[0]
                        class_confidences = {}
                        for idx, score in enumerate(confidence_scores):
                            if score > 0.5:
                                class_confidences[idx] = score
                        sorted_classes = sorted(class_confidences.items(), key=lambda x: x[1], reverse=True)
                    else:
                        sorted_classes = [(idx, 1.0) for idx in predicted_indices]
                except:
                    sorted_classes = [(idx, None) for idx in predicted_indices]
            
            else:  # BERT-based models
                # Tokenize text
                tokenizer = models["tokenizer"]
                encoding = tokenizer(
                    preprocessed_text, 
                    padding='max_length',
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Get predictions
                with torch.no_grad():
                    try:
                        outputs = models[selected_model](input_ids, attention_mask)
                        
                        # Use lower threshold for BERT models
                        threshold = 0.3
                        predictions = (outputs > threshold).int().cpu().numpy()[0]
                        confidence_scores = outputs.cpu().numpy()[0]
                        
                        # If no predictions with the threshold, take the top 3 categories
                        if np.sum(predictions) == 0:
                            top_indices = np.argsort(confidence_scores)[-3:][::-1]
                            new_predictions = np.zeros_like(predictions)
                            new_predictions[top_indices] = 1
                            predictions = new_predictions
                        
                        # Get class indices where predictions are 1
                        predicted_indices = np.where(predictions == 1)[0]
                        
                        # Map confidence scores to class indices
                        class_confidences = {}
                        for idx in predicted_indices:
                            class_confidences[idx] = float(confidence_scores[idx])
                        
                        # Sort by confidence
                        sorted_classes = sorted(class_confidences.items(), key=lambda x: x[1], reverse=True)
                    except:
                        sorted_classes = [(0, 0.5)]
            
            # Display results - Show only the top category with link
            st.markdown("### Classification Results")
            
            if len(sorted_classes) > 0:
                # Get the top category (highest confidence)
                top_idx, top_confidence = sorted_classes[0]
                
                # Create EUROVOC link for the category
                category_link = get_eurovoc_link(int(top_idx))
                
                # Display the top category with a link
                st.markdown(
                    f"<div class='top-category'>"
                    f"<div class='category-name'>"
                    f"<a href='{category_link}' target='_blank' style='text-decoration: none; color: #1E88E5;'>"
                    f"EUROVOC Category {top_idx} <span style='font-size: 0.8rem;'>&#128279;</span>"
                    f"</a></div>"
                    f"<div class='confidence-score'>Confidence: {top_confidence:.3f}</div>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                
                # Option to show all categories
                if st.checkbox("Show all detected categories"):
                    for idx, confidence in sorted_classes:
                        category_link = get_eurovoc_link(int(idx))
                        st.markdown(
                            f"<a href='{category_link}' target='_blank'>EUROVOC Category {idx}</a>: {confidence:.3f}",
                            unsafe_allow_html=True
                        )
            else:
                st.warning("No categories were identified for this petition.")

# Add a sample text option
st.markdown("### Sample Text")
if st.button("Load Sample Petition"):
    sample_text = """
    Petition on Environmental Protection Measures for the Baltic Sea Region
    To the European Parliament and the Commission of the European Union,
    We, the undersigned citizens of the European Union, are deeply concerned about the deteriorating state of the Baltic Sea ecosystem. The Baltic Sea faces numerous environmental challenges including eutrophication, chemical pollution, overfishing, and habitat destruction.
    Scientific studies have shown alarming levels of phosphorus and nitrogen from agricultural runoff, causing widespread algal blooms and oxygen-depleted dead zones. Additionally, persistent organic pollutants continue to pose serious threats to marine wildlife and potentially to human health.
    We therefore petition the European Parliament and Commission to:
    1. Strengthen existing regulations on agricultural practices in the Baltic Sea catchment area, with specific focus on fertilizer management and livestock farming.
    2. Increase funding for wastewater treatment facilities in all Baltic countries to meet highest standards of operation.
    3. Establish stricter monitoring of industrial discharge with enhanced penalties for violations.
    4. Develop a comprehensive ecosystem-based management plan for Baltic Sea fisheries.
    5. Create a dedicated Baltic Sea Restoration Fund to support innovative clean-up technologies and habitat restoration projects.
    We firmly believe that coordinated action at the European level is essential to preserve this shared natural heritage for future generations. The Baltic Sea is not only a vital economic resource but also a cultural and ecological treasure that deserves our immediate attention and protection.
    """
    st.session_state.petition_text = sample_text
    st.rerun()