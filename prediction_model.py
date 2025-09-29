from nltk.corpus import stopwords
import joblib
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from langdetect import detect

ps = PorterStemmer()
nltk.data.path.append('./nltk_data')


model = joblib.load('model2.pkl')
print('=> Pickle Loaded : Model ')
tfidfvect = joblib.load('tfidfvect2.pkl')
print('=> Pickle Loaded : Vectorizer')


class PredictionModel:
    output = {}

    # constructor
    def __init__(self, original_text):
        self.output['original'] = original_text
        self.output['language'] = None
        self.output['top_tokens'] = []


    # predict
    def predict(self):
        review = self.preprocess()
        text_vect = tfidfvect.transform([review]).toarray()
        pred = model.predict(text_vect)
        self.output['prediction'] = 'FAKE' if pred == 0 else 'REAL'
        # simple explainability: show top contributing tokens by absolute tf-idf values
        try:
            # get non-zero indices and sort by value desc
            vector = text_vect[0]
            indices = vector.nonzero()[0]
            values = [(i, vector[i]) for i in indices]
            values.sort(key=lambda x: x[1], reverse=True)
            feature_names = getattr(tfidfvect, 'get_feature_names_out', tfidfvect.get_feature_names)()
            self.output['top_tokens'] = [feature_names[i] for i, _ in values[:10]]
        except Exception:
            self.output['top_tokens'] = []
        return self.output


    # Helper methods
    def preprocess(self):
        text = self.output['original'] or ''
        # language detection (best-effort)
        try:
            lang = detect(text)
        except Exception:
            lang = 'en'
        self.output['language'] = lang
        # only English model is trained; if not English, keep alphas but do not stem
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        tokens = review.split()
        if lang == 'en':
            stop_words = set(stopwords.words('english'))
            tokens = [ps.stem(word) for word in tokens if word and word not in stop_words]
        else:
            # basic filtering for non-English
            tokens = [word for word in tokens if len(word) > 2]
        result = ' '.join(tokens)
        self.output['preprocessed'] = result
        return result
