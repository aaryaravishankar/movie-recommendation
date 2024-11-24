import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# reading the data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# combine them
combined_df = pd.merge(ratings, movies, on='movieId')
print(combined_df.head())

# recommendation system based on popularity


def popularity_based_recommendation(df, genre, min_ratings, num_rec):

    # Step 1: Filter by genre
    filter_by_genre = df[df['genres'].str.contains(
        genre, case=False, na=False)]

    # Step 2: Aggregate data at the movie level
    movie_stats = (
        filter_by_genre
        .groupby(['movieId', 'title'])
        .agg(
            num_reviews=('rating', 'count'),  # Count the number of reviews
            avg_rating=('rating', 'mean')    # Calculate the average rating
        )
        .reset_index()
    )

    movie_stats['num_reviews'] = pd.to_numeric(
        movie_stats['num_reviews'], errors='coerce')

    # Step 3: Filter by minimum reviews threshold
    recommendation = movie_stats[movie_stats['num_reviews'] >= min_ratings]

    # Step 4: Sort in descending order by average rating
    recommendation = recommendation.sort_values(
        by='avg_rating', ascending=False)

    # Step 5: Add serial numbers
    recommendation = recommendation.reset_index(drop=True)
    recommendation['S.No'] = recommendation.index + 1

    recommendation = recommendation[[
        'S.No', 'title', 'avg_rating', 'num_reviews']]
    # Step 6: Display the top 30 recommendations
    return recommendation.head(num_rec)


def content_based_recommendation(df, movie, num_rec):
    # find the genrs of selected movie
    filter = df['title'].str.contains(movie, case=False)
    selected_movie = df[filter]

    selected_genre = selected_movie['genres'].iloc[0].split('|')

    # compare simalirity with other movies
    def genre_similarity(genres):
        genre_list = genres.split('|')
        return len(set(selected_genre) & set(genre_list))

    df['similarity'] = df['genres'].apply(genre_similarity)

    rec = df[df['title'] != selected_movie['title'].iloc[0]
             ].sort_values(by='similarity', ascending=False)
    rec = rec.drop_duplicates(subset=['title'])
    rec = rec.reset_index(drop=True)
    rec['S.No'] = rec.index+1

    return rec[['S.No', 'title', 'genres']].head(num_rec)


def collaborative_based_recommendation(df, user_id, n, k):

    # Create user-item matrix
    user_item_matrix = df.pivot_table(
        index='userId', columns='movieId', values='rating').fillna(0)

    # Compute cosine similarity between users
    cosine_sim = cosine_similarity(user_item_matrix)
    user_similarity = pd.DataFrame(
        cosine_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Ensure the user_id exists
    if user_id not in user_similarity.index:
        return pd.DataFrame({"Error": [f"User ID {user_id} not found in the dataset."]})

    # Find top K similar users (excluding the target user itself)
    similar_users = user_similarity[user_id].sort_values(
        ascending=False).iloc[1:k + 1].index

    # Collect all movies rated by the similar users
    similar_users_ratings = user_item_matrix.loc[similar_users]

    # Get average rating for each movie rated by similar users
    avg_movie_ratings = similar_users_ratings.mean(axis=0)

    # Filter out movies already rated by the target user
    user_rated_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    avg_movie_ratings = avg_movie_ratings.drop(
        user_rated_movies, errors='ignore')

    # Sort by rating and select top N
    top_movies = avg_movie_ratings.sort_values(ascending=False).head(n)

    # Map movieId to titles
    movie_titles = df[['movieId', 'title']
                      ].drop_duplicates().set_index('movieId')
    recommended_movies = pd.DataFrame({
        'movieId': top_movies.index,
        'score': top_movies.values
    }).merge(movie_titles, on='movieId')

    # Add S.No for display
    recommended_movies.index += 1
    recommended_movies.reset_index(inplace=True)
    recommended_movies.rename(columns={'index': 'S.No'}, inplace=True)

    return recommended_movies[['S.No', 'title', 'score']]


# streamlit app
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎥",
    layout="wide",  # Optional: Use 'wide' layout for more space
    initial_sidebar_state="expanded"  # Sidebar is expanded by default
)

st.title("Movie Recommendation System")

# sidebar to select the recommendation type
st.sidebar.header("Choose Recommendation Type")
rec_type = st.sidebar.selectbox("Recommendation Type", [
                                'Popularity Based', 'Content Based', 'Collaborative Based'])

if rec_type == 'Popularity Based':
    st.header("Popularity Based Recommendations")
    genre = st.text_input("Enter a Movie Genre", value='Comedy')
    num_rec = st.text_input("Enter Number of Recommendations")
    min_ratings = st.text_input("Enter the minimum number of reviews")
    num_rec = int(num_rec) if num_rec.isdigit() else 10
    min_ratings = int(min_ratings) if min_ratings.isdigit() else 5

    # action
    if st.button("Get Recommendation"):
        rec = popularity_based_recommendation(
            combined_df, genre, min_ratings, num_rec)

        if rec is not None:
            st.write(f' Recommendations for {genre} :')
            st.table(rec)
        else:
            st.write(f'No recommendations found for {genre}')
elif rec_type == 'Content Based':
    st.header("Content-Based Recommendations")
    # inputs
    movie_name = st.text_input("Enter thr Movie Name")
    num_rec = st.slider('Number of recommendations', 1, 5, 10)

    if st.button('Get Recommendations'):
        rec = content_based_recommendation(combined_df, movie_name, num_rec)

        if rec is not None:
            st.write(f' Recommendations for {movie_name}:')
            st.table(rec)
        else:
            st.write(f'No recmmendations found fr the movie {movie_name}')
elif rec_type == 'Collaborative Based':
    st.header("Collaborative Filtering Recommendations")
    userID = st.text_input("Enter User ID")
    k = st.text_input("Number of Similar Users K")
    n = st.text_input("Number of Recommendations")

    userID = int(userID) if userID.isdigit() else None
    k = int(k) if k.isdigit() else 1
    n = int(n) if n.isdigit() else 5

    if st.button('Get Recommendation'):
        rec = collaborative_based_recommendation(combined_df, userID, n, k)
        if rec is not None:
            st.write("### Recommendations for User", userID)
            st.table(rec)
        else:
            st.write("No recommendations available.")
