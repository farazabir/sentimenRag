
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def get_sentiment(text: str) -> str:
    prompt = f"Analyze the sentiment of the following text and respond with one word (Positive, Negative, or Neutral):\n\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip()
