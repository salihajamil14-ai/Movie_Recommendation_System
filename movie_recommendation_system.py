import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_matrix(df):
    user_mapper = {uid: i for i, uid in enumerate(df['userId'].unique())}
    movie_mapper = {mid: i for i, mid in enumerate(df['movieId'].unique())}
    movie_inv_mapper = {i: mid for mid, i in movie_mapper.items()}

    user_index = df['userId'].map(user_mapper)
    movie_index = df['movieId'].map(movie_mapper)

    X = csr_matrix((df["rating"], (movie_index, user_index)),
                   shape=(len(movie_mapper), len(user_mapper)))
    
    return X, movie_mapper, movie_inv_mapper


def recommend_similar(movie_title, df, X, movie_mapper, movie_inv_mapper, k=5):
    movie_data = df[df['title'] == movie_title]
    if movie_data.empty:
        print(f"\nError: Movie '{movie_title}' not found in the dataset.")
        print("Please check the spelling or try another movie.")
        return []
    
    movie_id = movie_data['movieId'].iloc[0]
    
    if movie_id not in movie_mapper:
        print(f"\nError: Movie ID {movie_id} not found in the matrix.")
        return []
    
    movie_idx = movie_mapper[movie_id]
    movie_vec = X[movie_idx]

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(X)
    distances, indices = model.kneighbors(movie_vec, n_neighbors=k + 1)

    neighbor_ids = [movie_inv_mapper[i] for i in indices.flatten()[1:]]
    recommendations = df[df['movieId'].isin(neighbor_ids)]['title'].unique()

    print(f"\n{'='*60}")
    print(f"Because you liked: {movie_title}")
    print(f"{'='*60}")
    print("You might also enjoy:\n")
    for i, rec in enumerate(recommendations[:k], 1):
        print(f"   {i}. {rec}")
    print(f"{'='*60}\n")
    
    return list(recommendations[:k])


def load_and_prepare_data(ratings_file='ratings.csv', movies_file=None):
    try:
        ratings = pd.read_csv(ratings_file)
        print(f"Loaded {len(ratings):,} ratings from {ratings_file}")
        print(f"   Dataset shape: {ratings.shape}")
        print(f"\n   First few rows:")
        print(ratings.head())
        
        if movies_file and 'title' not in ratings.columns:
            if os.path.exists(movies_file):
                movies = pd.read_csv(movies_file)
                ratings = ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')
                print(f"\nMerged with movies dataset. Total unique movies: {ratings['movieId'].nunique()}")
            else:
                print(f"\nWarning: {movies_file} not found. Continuing without movie titles.")
        elif movies_file and 'title' in ratings.columns:
            print(f"\nTitle column already exists in ratings. Skipping merge.")
        
        return ratings
        
    except FileNotFoundError:
        print(f"\nError: File '{ratings_file}' not found.")
        print("   Please make sure the ratings.csv file is in the current directory.")
        return None
    except Exception as e:
        print(f"\nError loading data: {e}")
        return None


def display_statistics(df):
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"   Total ratings:      {len(df):,}")
    print(f"   Unique users:       {df['userId'].nunique():,}")
    print(f"   Unique movies:      {df['movieId'].nunique():,}")
    print(f"   Average rating:     {df['rating'].mean():.2f}")
    print(f"   Rating range:       {df['rating'].min()} - {df['rating'].max()}")
    print(f"\n   Rating distribution:")
    rating_dist = df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = (count / len(df)) * 100
        print(f"      {rating:>4}: {count:>6,} ({percentage:>5.1f}%)")
    print("="*60)


def main():
    print("\n" + "="*60)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("="*60)
    print("   Using collaborative filtering with cosine similarity")
    print("="*60)
    
    ratings = None
    if os.path.exists('movies.csv'):
        ratings = load_and_prepare_data('ratings.csv', 'movies.csv')
    else:
        ratings = load_and_prepare_data('ratings.csv')
    
    if ratings is None:
        print("\nError: Please ensure you have a ratings.csv file.")
        print("   The file should have columns: userId, movieId, rating")
        print("   Optionally, you can also have movies.csv with columns: movieId, title")
        return
    
    if 'title' not in ratings.columns:
        print("\nWarning: 'title' column not found in dataset.")
        print("   Please ensure your dataset has a 'title' column or provide a movies.csv file.")
        return
    
    display_statistics(ratings)
    
    print("\n" + "="*60)
    print("CREATING USER-ITEM MATRIX")
    print("="*60)
    X, movie_mapper, movie_inv_mapper = create_matrix(ratings)
    print(f"   Matrix shape: {X.shape[0]:,} movies × {X.shape[1]:,} users")
    sparsity = (1 - X.nnz / (X.shape[0] * X.shape[1])) * 100
    print(f"   Matrix sparsity: {sparsity:.2f}%")
    
    user_item_matrix = ratings.pivot_table(
        index="title", columns="userId", values="rating"
    )
    print(f"   Pivot table shape: {user_item_matrix.shape}")
    print(f"\n   Sample of user-item matrix (first 5 rows, first 5 columns):")
    print(user_item_matrix.iloc[:5, :5])
    
    print("\n" + "="*60)
    print("GETTING RECOMMENDATIONS")
    print("="*60)
    
    sample_movies = ratings['title'].unique()[:5]
    print(f"\n   Sample movies in dataset:")
    for i, movie in enumerate(sample_movies, 1):
        print(f"      {i}. {movie}")
    
    if len(sample_movies) > 0:
        test_movie = sample_movies[0]
        print(f"\n   Testing recommendation for: {test_movie}")
        recommendations = recommend_similar(
            test_movie, ratings, X, movie_mapper, movie_inv_mapper, k=5
        )
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("   Enter movie titles to get recommendations")
    print("   Type 'quit', 'exit', or 'q' to exit")
    print("="*60)
    
    while True:
        try:
            movie_title = input("\nEnter a movie title: ").strip()
            
            if movie_title.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Movie Recommendation System!")
                print("   Goodbye!\n")
                break
            
            if not movie_title:
                print("   Warning: Please enter a valid movie title.")
                continue
            
            recommend_similar(
                movie_title, ratings, X, movie_mapper, movie_inv_mapper, k=5
            )
            
        except KeyboardInterrupt:
            print("\n\nThank you for using the Movie Recommendation System!")
            print("   Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError: An error occurred: {e}")
            print("   Please try again.")


if __name__ == "__main__":
    main()
