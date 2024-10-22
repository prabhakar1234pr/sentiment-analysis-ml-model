

import kaggle
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting the dataset

import tensorflow as tf  # Ensure you're using TensorFlow correctly
from tensorflow.keras.models import Sequential  # For building a sequential model
from tensorflow.keras.layers import Dense, Embedding, LSTM  # LSTM layers for the model
from tensorflow.keras.preprocessing.text import Tokenizer  # For text tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences
import os  # Standard library for operating system functionality
import json  # Standard library for JSON handling
from zipfile import ZipFile


# Optional: Suppress warnings (not necessary)
import warnings
warnings.filterwarnings('ignore')
kaggle_dictionary = json.load(open("kaggle.json"))
os.environ['KAGGLE_CONFIG_DIR'] = '/app/.kaggle'
os.environ["KAGGLE_USERNAME"] = kaggle_dictionary['username']
os.environ["KAGGLE_KEY"] = kaggle_dictionary['key']
import pandas as pd
def setup_kaggle_credentials():
    
    os.environ['KAGGLE_USERNAME'] = st.secrets["kaggle"]["username"]
    os.environ['KAGGLE_KEY'] = st.secrets["kaggle"]["key"]
def download_kaggle_dataset():
    # Set up Kaggle API credentials
    api = KaggleApi()
    api.authenticate()

    # Download the dataset (you may need to change the owner/dataset name to match the Kaggle dataset you're using)
    dataset_name = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    
    # Download dataset to a specified path
    api.dataset_download_files(dataset_name, path=".", unzip=True)

# Check if dataset is already downloaded; if not, download it
if not os.path.exists('IMDB Dataset.csv'):
    download_kaggle_dataset()

# Load and display dataset
df = pd.read_csv('IMDB Dataset.csv')

df.replace({"sentiment":{"positive":1,"negative":0}},inplace = True)
train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(train_data['review'])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['review']),maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['review']),maxlen=200)
Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]
#create a model
model = Sequential()
model.add(Embedding(input_dim=5000,output_dim=128,input_length = 200 ,input_shape=(200,)))
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train,Y_train,batch_size=65,epochs=5,validation_split=0.2)
model.save('my_movie_review_model.keras')
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
def predict_sentiment(review):
    # Tokenize and pad the review
    review_sequence = tokenizer.texts_to_sequences([review])
    pad_review_sequence = pad_sequences(review_sequence, maxlen=200)

    # Make a prediction
    prediction = model.predict(pad_review_sequence)

    # Interpret the prediction
    if prediction > 0.5:
        return "Positive"
    else:
        return "Negative"

new_review = input("Enter Your review here: ")
predicted_sentiment = predict_sentiment(new_review)
print(f"Predicted Sentiment: {predicted_sentiment}")
import streamlit as st

model = tf.keras.models.load_model('my_movie_review_model.keras')  

# Streamlit app structure
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")

# User input for the review
user_input = st.text_input("Enter your review here:")

# Perform prediction if there's an input
if user_input:
    prediction = predict_sentiment(user_input)
    st.write(f"Predicted Sentiment: {prediction}")


