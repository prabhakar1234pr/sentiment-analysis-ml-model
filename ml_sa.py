import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# Load and preprocess dataset
df = pd.read_csv("IMDB Dataset.csv")
df.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['review'])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['review']), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['review']), maxlen=200)
Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]

# Create the model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, Y_train, batch_size=64, epochs=5, validation_split=0.2)
model.save('my_movie_review_model.keras')

# Prediction function
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_movie_review_model.keras')

def predict_sentiment(review):
    # Tokenize and pad the review
    review_sequence = tokenizer.texts_to_sequences([review])
    pad_review_sequence = pad_sequences(review_sequence, maxlen=200)

    # Make a prediction
    prediction = model.predict(pad_review_sequence)

    # Interpret the prediction
    return "Positive" if prediction > 0.5 else "Negative"

# Streamlit app
st.title("IMDB Movie Reviews Sentiment Analysis")
new_review = st.text_input("Enter Your review here:")

if new_review:
    model = load_model()
    predicted_sentiment = predict_sentiment(new_review)
    st.write(f"Predicted Sentiment: {predicted_sentiment}")
