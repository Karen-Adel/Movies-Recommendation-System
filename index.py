import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import ast
import nltk
import pickle
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

# Load datasets
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

app = Flask(__name__)

# Merge datasets on title
movies = movies.merge(credits, on='title')

# Display column names
# print(movies.columns)

# Select relevant columns
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Check for missing values and drop them
# print(movies.isnull().sum())
movies.dropna(inplace=True)

# Check for duplicates
print(f"Number of duplicate rows: {movies.duplicated().sum()}")

# Function to extract names from a JSON-like string
def convert(text):
    if pd.isna(text):
        return []
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except (ValueError, SyntaxError):
        return []

# Function to extract top 3 cast members
def convert_cast(text):
    if pd.isna(text):
        return []
    try:
        return [i['name'] for i in ast.literal_eval(text)[:3]]  # Take only top 3
    except (ValueError, SyntaxError):
        return []

# Function to extract director name
def find_director(text):
    if pd.isna(text):
        return []
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
    except (ValueError, SyntaxError):
        return []
    return []

# Function to remove spaces in names
def remove_space(word_list):
    return [word.replace(" ", "") for word in word_list]

#to make love as loving and loved the same word to appear while searching
ps=PorterStemmer()
def stems(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)


# Apply transformations
movies['genres'] = movies['genres'].apply(convert).apply(remove_space)
movies['keywords'] = movies['keywords'].apply(convert).apply(remove_space)
movies['cast'] = movies['cast'].apply(convert_cast).apply(remove_space)
movies['crew'] = movies['crew'].apply(find_director).apply(remove_space)

# Convert overview to a list of words
movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())

# Create a 'tags' column by combining all relevant text-based features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Keep only required columns
new_df = movies[['id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Display the first few rows
# print(new_df.head())
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
# print(new_df.iloc[0]['tags'])

new_df['tags'] = new_df['tags'].apply(stems)
# print(new_df.iloc[0]['tags'])

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
# print(vector)
similarity = cosine_similarity(vector)
# print(similarity)

def recommend_movie(movie):
    movie = movie.lower()  # Normalize input
    if movie not in new_df['title'].str.lower().values:
        print(f"Movie '{movie}' not found in dataset.")
        return

    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    print(f"Recommendations for '{movie.title()}':")
    for i in distances[1:6]:
        print(new_df.iloc[i[0]]['title'])
recommend_movie('avatar')



# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form.get('movie')
    recommendations = recommend_movie(movie)
    return render_template('recommendations.html', movie=movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)



pickle.dump(new_df, open('artifacts/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))

