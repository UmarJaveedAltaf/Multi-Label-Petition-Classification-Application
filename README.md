# 📝 Multi-Label Petition Classification Application

An end-to-end **Natural Language Processing (NLP)** application that classifies **European Union petition texts** into relevant **EUROVOC categories** using both **traditional machine learning** and **deep learning** models. The project includes a fully interactive **Streamlit web application** for real-time classification and model comparison.

---

## 🔍 Project Overview

- **Task:** Multi-label classification of EU petitions  
- **Dataset:** EURLEX (LexGLUE benchmark) from HuggingFace  
- **Labels:** EUROVOC concept categories  
- **Models:** TF-IDF + ML models and BERT-based DL models  
- **Deployment:** Streamlit Web Application  
- **Objective:** Compare model performance and demonstrate real-time legal text classification  

Each petition can belong to **multiple EUROVOC categories**, making this a challenging and realistic multi-label NLP task.

---

## 📊 Models Implemented

| Model | Description |
|-------|-------------|
| **TF-IDF + Naive Bayes** | Probabilistic baseline classifier |
| **TF-IDF + Passive Aggressive** | Strong linear classifier for large-scale text |
| **BERT + GRU** | Contextual embeddings with sequential modeling |
| **BERT + BiLSTM** | Bidirectional sequence modeling over BERT embeddings |

---

## 📈 Model Performance

Performance on the EURLEX dataset:

| Model | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| TF-IDF + Naive Bayes | 0.5566 | 0.7725 | 0.4350 |
| TF-IDF + Passive Aggressive | 0.7536 | 0.7821 | 0.7272 |
| **BERT + GRU** | **0.7585** | **0.8549** | **0.6817** |
| BERT + BiLSTM | 0.7238 | 0.8590 | 0.6254 |

**Best overall model:** **BERT + GRU**, achieving the highest F1 score with strong precision-recall balance.

---

## 🖥️ Streamlit Application Features

- 🔄 Model selection from sidebar  
- ✍️ Custom petition text input  
- 📊 Confidence-based prediction display  
- 🔗 Clickable EUROVOC category links  
- 🧪 Sample petition loader  
- 🌙 Clean, professional dark-mode UI  

---

## 📸 Application Screenshots

<table>
  <tr>
    <td><img src="screenshots/bert_gru.png" width="400" alt="BERT + GRU"/></td>
    <td><img src="screenshots/bert_bilstm.png" width="400" alt="BERT + BiLSTM"/></td>
  </tr>
  <tr>
    <td><img src="screenshots/passive_aggressive.png" width="400" alt="Passive Aggressive"/></td>
    <td><img src="screenshots/naive_bayes.png" width="400" alt="Naive Bayes"/></td>
  </tr>
</table>

---

## 📦 Model Files Setup

⚠️ **Important:** GitHub does not store large model files.

Before running the Streamlit application, ensure the following files are present in the project root directory:

```
tfidf_vectorizer.pkl
multilabel_binarizer.pkl
naive_bayes_model.pkl
passive_aggressive_model.pkl
bert_gru_model.pt
bert_bilstm_model.pt
```

### Exporting Models from Notebook

```python
import joblib
import torch

# Save TF-IDF vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Save MultiLabelBinarizer
joblib.dump(mlb, "multilabel_binarizer.pkl")

# Save traditional ML models
joblib.dump(model_nb, "naive_bayes_model.pkl")
joblib.dump(model_pa, "passive_aggressive_model.pkl")

# Save deep learning models
torch.save(model_bert_gru, "bert_gru_model.pt")
torch.save(model_bilstm, "bert_bilstm_model.pt")
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NLP_FINAL_PROJECT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present** (see Model Files Setup section)

4. **Run the Streamlit application**
   ```bash
   streamlit run streamlit-app.py
   ```

5. **Open in browser**
   ```
   http://localhost:8501
   ```

---

## 📁 Project Structure

```
NLP_FINAL_PROJECT/
├── codebase.ipynb              # Main notebook with model training
├── streamlit-app.py            # Streamlit web application
├── requirements.txt            # Python dependencies
├── *.pkl                       # Pickled ML models and vectorizers
├── *.pt                        # PyTorch deep learning models
├── README.md                   # Project documentation
└── screenshots/                # Application screenshots
    ├── bert_gru.png
    ├── bert_bilstm.png
    ├── passive_aggressive.png
    └── naive_bayes.png
```

---

## 🌐 EUROVOC Integration

Each predicted label links directly to the official EUROVOC concept page:

**Base URL:** `https://op.europa.eu/en/web/eu-vocabularies/eurovoc`

This allows users to explore the semantic meaning and context of each predicted category within the EU's multilingual thesaurus.

---

## 🛠️ Technologies Used

- **Machine Learning:** scikit-learn, TF-IDF vectorization
- **Deep Learning:** PyTorch, Transformers (BERT)
- **Web Framework:** Streamlit
- **Data Processing:** pandas, numpy
- **Dataset:** HuggingFace Datasets (LexGLUE/EURLEX)

---

## 📊 Dataset Information

- **Source:** LexGLUE benchmark (Legal-domain Language Understanding Evaluation)
- **Dataset:** EURLEX (European Union legal documents)
- **Task Type:** Multi-label text classification
- **Labels:** EUROVOC thesaurus categories
- **Language:** English

---
## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Umar Javeed Altaf**  
Graduate Student – Artificial Intelligence  
Northeastern University
---

## ⭐ Acknowledgments

- HuggingFace for the EURLEX dataset
- LexGLUE benchmark creators
- EU Publications Office for EUROVOC thesaurus
- Streamlit team for the excellent framework

---

**⚖️ Legal Text Classification Made Simple**
