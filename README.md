# Fake_News_Detection

This project is designed to classify news articles as real or fake using machine learning models. It includes both a web interface for real-time predictions via Streamlit and a model training pipeline for building the classifier.

Features
Real-time Fake News Detection: Users can input news articles and get a prediction of whether the article is real or fake.
Model Training: The project includes scripts to train a machine learning model on labeled news datasets (real and fake news).
TF-IDF Vectorizer: Text preprocessing includes tokenization, stopword removal, and stemming.
Streamlit Interface: A user-friendly interface for submitting articles and displaying results.
Requirements
You will need the following libraries to run this project:

streamlit
joblib
numpy
nltk
scikit-learn
pandas
matplotlib
seaborn
tensorflow
You can install the required packages by running:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn nltk tensorflow streamlit
Setup Instructions
1. Model Training
The model is trained using two datasets: one containing real news articles and another containing fake news. The training process involves the following steps:

Loading the datasets (True.csv and Fake.csv).
Preprocessing the text (lowercasing, stopword removal, stemming).
Vectorizing the text using TF-IDF.
Splitting the data into training and test sets.
Training a Logistic Regression model to classify articles.
After training, the model and vectorizer are saved as fake_news_model.pkl and tfidf_vectorizer.pkl, respectively, for later use in the Streamlit app.

2. Streamlit Web Interface
Run the Streamlit app with the following command:

bash
Copy
Edit
streamlit run app.py
You can then paste a news article into the input field, and the app will predict whether it's real or fake.

3. Model Files
The app requires two files to function properly:

fake_news_model.pkl: The trained machine learning model.
tfidf_vectorizer.pkl: The TF-IDF vectorizer used for text transformation.
Ensure these files are located in the specified directory or update the paths in the code accordingly.
