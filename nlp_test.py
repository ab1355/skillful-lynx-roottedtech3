import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def test_nlp():
    text = "Apple Inc. is planning to open a new store in New York City next month. The CEO, Tim Cook, expressed excitement about the expansion."

    # Tokenization
    tokens = word_tokenize(text)
    print("Tokens:", tokens)

    # Frequency Distribution
    fdist = FreqDist(tokens)
    print("Most common words:", fdist.most_common(5))

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
    print("Filtered tokens:", filtered_tokens)

test_nlp()