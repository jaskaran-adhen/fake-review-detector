import re

# Standard English stopwords to fall back on if NLTK is unavailable
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

_nltk_initialized = False
_stemmer = None
_lemmatizer = None

def init_nltk():
    """Safely initialize NLTK downloads and classes with fallback support."""
    global _nltk_initialized, _stemmer, _lemmatizer, STOPWORDS
    if _nltk_initialized:
        return
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        
        STOPWORDS = set(stopwords.words('english'))
        _stemmer = PorterStemmer()
    except Exception as e:
        # Fall back to standard set, print info warning
        print(f"[SYSTEM INFO] NLTK initialization fallback. Using built-in stopwords. Error: {e}")
    _nltk_initialized = True

def clean_text(text, use_stemming=True):
    """
    Preprocess raw review text:
    - Lowercase text
    - Remove punctuation and special characters
    - Remove numbers
    - Remove extra spaces
    - Remove stopwords
    - Apply word stemming
    """
    if not isinstance(text, str):
        return ""
    
    init_nltk()
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 3. Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # 4. Tokenize and remove extra spaces
    words = text.split()
    
    # 5. Remove stopwords
    words = [w for w in words if w not in STOPWORDS]
    
    # 6. Apply Stemming
    cleaned_words = []
    for w in words:
        if use_stemming:
            if _stemmer:
                cleaned_words.append(_stemmer.stem(w))
            else:
                # Basic suffix-stripping fallback stemmer
                if w.endswith('ing') and len(w) > 5:
                    w = w[:-3]
                elif w.endswith('ly') and len(w) > 4:
                    w = w[:-2]
                elif w.endswith('es') and len(w) > 4:
                    w = w[:-2]
                elif w.endswith('s') and len(w) > 3 and not w.endswith('ss'):
                    w = w[:-1]
                cleaned_words.append(w)
        else:
            cleaned_words.append(w)
            
    return " ".join(cleaned_words)
