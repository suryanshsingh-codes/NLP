import streamlit as st
import pandas as pd
import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# ============================================
# LOAD SAVED MODELS
# ============================================

model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
emotion_mapping = joblib.load('emotion_mapping.pkl')

reverse_mapping = {v: k for k, v in emotion_mapping.items()}

# ============================================
# TEXT CLEANING FUNCTION
# ============================================

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    new_text = ""
    for char in text:
        if not char.isdigit():
            new_text += char
    text = new_text
    
    new_text = ""
    for char in text:
        if char.isascii():
            new_text += char
    text = new_text
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return ' '.join(cleaned_words)

# ============================================
# PREDICTION
# ============================================

def predict_emotion(text):
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    emotion_name = reverse_mapping[prediction]
    return emotion_name, probabilities, cleaned_text

# ============================================
# UI
# ============================================

st.set_page_config(page_title="Check Your Emojion", page_icon="😊")
st.title("😊 My Emotion")
st.markdown("Enter text to predict the emotion!")

user_input = st.text_area("Your text:", height=150)

if st.button("Predict"):
    if user_input.strip():
        emotion, probs, cleaned = predict_emotion(user_input)
        
        st.success(f"### Predicted Emotion: **{emotion}**")
        st.metric("Confidence", f"{max(probs)*100:.1f}%")
        
        prob_df = pd.DataFrame({
            'Emotion': list(reverse_mapping.values()),
            'Probability': probs
        }).sort_values('Probability', ascending=False)
        
        st.bar_chart(prob_df.set_index('Emotion'))
        st.dataframe(prob_df)
    else:
        st.warning("Please enter some text!")

st.sidebar.markdown("### Emotions")
for num, name in reverse_mapping.items():
    st.sidebar.write(f"{num}: {name}")