import pandas as pd
import numpy as np
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st

# Load dataset
movies = pd.read_csv("data/tmdb_5000_movies.csv")

# normalize movie titles
def normalize_title(title):
    return re.sub(r'[^a-zA-Z0-9]', '', title).lower()

# Ensure case-insensitive and special character insensitive title matching
movies['normalized_title'] = movies['title'].apply(normalize_title)

# Content-Based Filtering using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = vectorizer.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_movie_poster(title):
    """Fetch movie poster using TMDb API"""
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
    response = requests.get(search_url).json()
    if response['results']:
        poster_path = response['results'][0].get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

def recommend_movies_content(movie_title, num_recommendations=5):
    movie_title = normalize_title(movie_title)
    if movie_title not in movies['normalized_title'].values:
        return []
    idx = movies[movies['normalized_title'] == movie_title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    return [movies.iloc[i[0]]['title'] for i in scores]


with open("recommendation_model.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)

# Streamlit Web App
st.title("Movie Recommendation System")

# Autocomplete movie title
movie_title = st.text_input("Enter a movie title:")
suggested_titles = movies[movies['title'].str.contains(movie_title, case=False, na=False)]['title'].tolist()[:5]

if suggested_titles:
    st.write("Did you mean:")
    for title in suggested_titles:
        st.write(title)

if st.button("Get Recommendations"):
    # Display searched movie poster
    searched_poster = get_movie_poster(movie_title)
    if searched_poster:
        st.image(searched_poster, caption=movie_title, width=200)
    
    recommendations = recommend_movies_content(movie_title, 5)
    if recommendations:
        st.write("Recommended Movies:")
        for rec in recommendations:
            poster_url = get_movie_poster(rec)
            st.write(rec)
            if poster_url:
                st.image(poster_url, width=150)
    

