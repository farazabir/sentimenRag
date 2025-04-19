import streamlit as st
from utils.sentiment import get_sentiment
from utils.explain import explain_lime, explain_shap
import shap
from matplotlib import pyplot as plt 

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("Sentiment Analyzer (Gemini + LIME + SHAP)")

text_input = st.text_area("Enter your sentence or paragraph:")
if st.button("Analyze") and text_input.strip():
    with st.spinner("Getting sentiment from Gemini..."):
        sentiment = get_sentiment(text_input)
    st.success(f"Sentiment: **{sentiment}**")
    st.markdown("---")
    st.subheader("LIME Explanation")
    lime_fig = explain_lime(text_input)
    st.image(lime_fig, caption="LIME Explanation", use_container_width=True)
    
    st.subheader("SHAP Explanation (Fallback Model)")
    shap_values, feature_names = explain_shap(text_input)
    fig = plt.figure()
    shap.summary_plot(shap_values.values, feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Enter some text to start analysis")
