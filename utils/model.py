
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = ["I love this!", "I hate it.", "It’s okay."]
labels = ["Positive", "Negative", "Neutral"]


pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
pipeline.fit(texts, labels)

def get_pipeline():
    return pipeline
