import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

st.title("Naive Bayes Classifier Demonstrator")

st.write("""
    ### Simple Text Classification Using Naive Bayes
    This application demonstrates how the Naive Bayes classifier can be used to classify documents into different categories
    using a simple dataset of short text documents.
""")

# Sample data
data = {
    "Text": ["This is a great movie", "This is a terrible movie",
             "I love this film", "I hate this film",
             "This movie is fantastic", "This movie is awful"],
    "Category": ["Positive", "Negative", "Positive", "Negative", "Positive", "Negative"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the data
st.write("#### Sample Data")
st.dataframe(df)

# Text preprocessing and vectorization
vectorizer = CountVectorizer(stop_words='english')

# Sidebar options for model configuration
st.sidebar.header("Model Settings")
test_size = st.sidebar.slider("Test Size", 0.1, 0.9, 0.2, 0.1, help="Size of the test set.")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Category'], test_size=test_size, random_state=42)

# Vectorize the text data
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Display the matrix of word counts
st.write("#### Word Count Matrix")
word_count_df = pd.DataFrame(X_train_counts.toarray(), columns=vectorizer.get_feature_names_out(), index=[f'Train {i}' for i in range(1, len(X_train) + 1)])
st.dataframe(word_count_df)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Predictions and their probabilities
predictions = clf.predict(X_test_counts)
prediction_probabilities = clf.predict_proba(X_test_counts)

st.write("#### Predictions and Probabilities on Test Data")
for i, (doc, category) in enumerate(zip(X_test, predictions)):
    st.write(f"Document: {doc}")
    st.write(f"Predicted Category: {category}")
    probs = {clf.classes_[i]: prob for i, prob in enumerate(prediction_probabilities[i])}
    st.write(f"Probabilities: {probs}")

# Show accuracy
accuracy = clf.score(X_test_counts, y_test)
st.write(f"#### Accuracy: {accuracy:.2f}")

st.write("""
    ### How it Works
    - **Text Preprocessing:** Documents are converted into a matrix of token counts, creating a bag-of-words model.
    - **Model Training:** A Naive Bayes classifier is trained on the vectorized text data.
    - **Prediction:** The classifier predicts the category for new documents based on learned patterns, with probabilities indicating confidence in predictions.
    - **Accuracy:** Reflects the proportion of test documents correctly classified by the model.
""")
