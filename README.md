# ✨ SK Hashtag Predictor

An AI-powered hashtag generator designed to analyze post content and suggest the most relevant tags. This project utilizes a custom **NLP Pipeline** and **Multi-label Classification** to bridge the gap between raw text and social media engagement.

---

## 🚀 Project Overview
- **Language:** Python
- **Dataset:** Built using **Web Scraping** for real-world relevance.
- **Models:** Multi-model consensus (SVC, Naive Bayes, Logistic Regression).
- **Interface:** Interactive Desktop GUI using **Tkinter**.

---

## 🏗️ The NLP Pipeline
The project follows a structured Natural Language Processing pipeline to ensure high accuracy and clean feature extraction.

### 1. Advanced Text Preprocessing
Raw text is cleaned through several stages:
* **Lowercasing:** Standardizing text to prevent case-sensitive duplicates.
* **Noise Removal:** Removing URLs, @mentions, and Emojis using Regular Expressions.
* **Spell Correction:** Ensuring technical and common terms are correctly recognized.
* **Standardization:** Stripping punctuation and extra whitespace for a clean input.

### 2. Feature Extraction: TF-IDF
We use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors.
> **Purpose:** TF-IDF evaluates how important a word is to a document relative to a collection of documents. Unlike simple word counts, it penalizes common "stop words" (like *the*, *is*) and rewards unique keywords, ensuring the model focuses on the words that actually define the specific topic.

### 3. Label Handling: MultiLabelBinarizer
Since a single post can belong to multiple categories (e.g., `#AI` and `#FutureTech`), we use the **MultiLabelBinarizer**. This transforms the target hashtags into a binary format suitable for multi-label classification.

---

## 🧠 Machine Learning Architecture
We trained three high-performance models using the **OneVsRestClassifier** strategy:

1.  **LinearSVC (Support Vector Classifier):**
    * **The Advantage:** LinearSVC is highly effective in high-dimensional spaces (which is exactly what text data is). It works by finding the optimal hyperplane that maximizes the margin between different classes. For this project, it provides the most robust boundary for complex, overlapping hashtags.
2.  **Multinomial Naive Bayes:**
    * A probabilistic learning algorithm that is particularly fast and effective for text classification based on word frequencies.
3.  **Logistic Regression:**
    * Used to provide a strong statistical baseline and reliable probability scores for the predicted tags.

---

## 🎨 Interactive GUI Features
The project features a modern, user-friendly interface:
* **Visual Hierarchy:** Large, **Bold** fonts for both input and output to ensure readability.
* **Sky Blue Aesthetic:** A professional "Light Mode" theme with interactive hover effects on buttons.
* **Smart Actions:** Includes a **Copy to Clipboard** button and a **Reset** function for a seamless workflow.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ihsanullah26/SK-Hashtag_Predictor.git](https://github.com/ihsanullah26/SK-Hashtag_Predictor.git)
   cd SK-Hashtag_Predictor
2. **Install The Required Library:**
   ```bash
   pip install joblib scikit-learn nltk pyperclip emoji numpy
1. **Run The App:**
   ```bash
   python3 app.py
1. **Interact With GUI:**
   ```bash
   python3 gui.py
