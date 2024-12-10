import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer once
with open(r'tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Prediction function
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_movie_review_model.keras')

def predict_sentiment(review, model, tokenizer):
    # Tokenize and pad the review
    review_sequence = tokenizer.texts_to_sequences([review])
    pad_review_sequence = pad_sequences(review_sequence, maxlen=200)

    # Make a prediction
    prediction = model.predict(pad_review_sequence)

    # Interpret the prediction
    return "Positive" if prediction > 0.5 else "Negative"

# Streamlit app
st.title("IMDB Movie Reviews Sentiment Analysis")
new_review = st.text_input("Enter your review here:")

if new_review:
    model = load_model()  # Load the model once
    predicted_sentiment = predict_sentiment(new_review, model, tokenizer)
    st.write(f"Predicted Sentiment: {predicted_sentiment}")

