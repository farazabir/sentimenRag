from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import shap
import io

from .model import get_pipeline

pipeline = get_pipeline()
lime_explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])

def explain_lime(text: str):
    exp = lime_explainer.explain_instance(text, pipeline.predict_proba, num_features=8)
    fig = exp.as_pyplot_figure()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf
def explain_shap(text: str):
    vectorizer = pipeline.named_steps['tfidfvectorizer']
    model = pipeline.named_steps['logisticregression']
    feature_names = vectorizer.get_feature_names_out()
    text_vector = vectorizer.transform([text])
    explainer = shap.Explainer(model, vectorizer.transform(["I love this!", "I hate it.", "It's okay."]))
    shap_values = explainer(text_vector)
    return shap_values, feature_names
