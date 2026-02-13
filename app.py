import pandas as pd
import re, emoji, string, joblib, random
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.corpus import words, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# --- STEP 1: INITIALIZE ---
nltk.download('words')
nltk.download('wordnet') # Needed for synonyms
nltk.download('omw-1.4')
ENGLISH_DICT = set(w.lower() for w in words.words())

# --- STEP 2: SYNTHETIC DATA GENERATOR ---

def get_synonyms(word):
    """Finds synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace('_', ' ').replace('-', ' ').lower()
            if synonym != word and synonym in ENGLISH_DICT:
                synonyms.add(synonym)
    return list(synonyms)

def augment_text(text, n=1):
    """Creates a synthetic version of the text by replacing words with synonyms."""
    words_list = text.split()
    if len(words_list) < 3: return text # Don't augment very short text
    
    new_text = words_list.copy()
    words_to_replace = random.sample(range(len(all_words := [w for w in words_list if len(w) > 3])), min(len(all_words), n))
    
    for idx in words_to_replace:
        word = words_list[idx]
        syns = get_synonyms(word)
        if syns:
            new_text[idx] = random.choice(syns)
            
    return " ".join(new_text)

# --- STEP 3: PREPROCESSING & BALANCING ---

def strict_clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|@\S+|\d+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split()) # Keep all words, don't filter against a dictionary

def process_and_augment(df, target_count=15):
    """
    Identifies hashtags with few samples and generates 
    synthetic data to reach the target_count.
    """
    augmented_records = []
    
    # Track counts per hashtag
    all_tags = [tag for sublist in df['htags'] for tag in sublist]
    tag_counts = Counter(all_tags)
    
    for _, row in df.iterrows():
        # Check if this tweet contains a "rare" hashtag
        is_rare = any(tag_counts[tag] < target_count for tag in row['htags'])
        
        if is_rare:
            # Create 2 synthetic versions of this tweet
            for _ in range(2):
                new_text = augment_text(row['ttext'])
                if new_text != row['ttext']:
                    augmented_records.append({'ttext': new_text, 'htags': row['htags']})
                    
    return pd.concat([df, pd.DataFrame(augmented_records)], ignore_index=True)

# --- STEP 4: MAIN PIPELINE ---

FILE_NAME = 'final_dataset.csv' 
raw_df = pd.read_csv(FILE_NAME)

# Initial cleaning
df = pd.DataFrame()
df['htags'] = raw_df['htags'].apply(lambda x: re.findall(r'#\w+', str(x).lower()))
df['ttext'] = raw_df['ttext'].apply(strict_clean)
df = df[(df['htags'].map(len) > 0) & (df['ttext'].str.strip() != '')].copy()

print(f"Original size: {len(df)}")

# Apply Synthetic Augmentation
df = process_and_augment(df, target_count=15)

print(f"Size after synthetic augmentation: {len(df)}")

# --- STEP 5: TRAINING ---

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
X = tfidf.fit_transform(df['ttext'])

bln = MultiLabelBinarizer()
y = bln.fit_transform(df['htags'])

# Expert Models
svc = OneVsRestClassifier(LinearSVC(dual='auto', class_weight='balanced'))
nb = OneVsRestClassifier(MultinomialNB())
lr = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced'))

svc.fit(X, y)
nb.fit(X, y)
lr.fit(X, y)

# Save
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(bln, 'binarizer.pkl')
joblib.dump(svc, 'model_svc.pkl')
joblib.dump(nb, 'model_nb.pkl')
joblib.dump(lr, 'model_lr.pkl')

print("Models trained with synthetic data and saved.")