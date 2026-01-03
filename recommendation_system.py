import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

def create_matrix(df):
    """
    Create a sparse user-item matrix from ratings dataframe.
    
    Parameters:
    df (DataFrame): DataFrame with columns: userId, movieId, rating
    
    Returns:
    X: Sparse matrix (movies x users)
    movie_mapper: Dictionary mapping movieId to matrix index
    movie_inv_mapper: Dictionary mapping matrix index to movieId
    """
    user_mapper = {uid: i for i, uid in enumerate(df['userId'].unique())}
    movie_mapper = {mid: i for i, mid in enumerate(df['movieId'].unique())}
    movie_inv_mapper = {i: mid for mid, i in movie_mapper.items()}

    user_index = df['userId'].map(user_mapper)
    movie_index = df['movieId'].map(movie_mapper)

    X = csr_matrix((df["rating"], (movie_index, user_index)),
                   shape=(len(movie_mapper), len(user_mapper)))
    return X, movie_mapper, movie_inv_mapper


def recommend_similar(movie_title, df, X, movie_mapper, movie_inv_mapper, k=5):
    """
    Recommend similar movies based on a given movie title.
    
    Parameters:
    movie_title (str): Title of the movie to find similar movies for
    df (DataFrame): DataFrame with movie ratings
    X: Sparse user-item matrix
    movie_mapper: Dictionary mapping movieId to matrix index
    movie_inv_mapper: Dictionary mapping matrix index to movieId
    k (int): Number of recommendations to return
    
    Returns:
    recommendations: List of recommended movie titles
    """
    # Check if movie exists in dataset
    movie_data = df[df['title'] == movie_title]
    if movie_data.empty:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return []
    
    movie_id = movie_data['movieId'].iloc[0]
    
    # Check if movie_id exists in mapper
    if movie_id not in movie_mapper:
        print(f"Movie ID {movie_id} not found in the matrix.")
        return []
    
    movie_idx = movie_mapper[movie_id]
    movie_vec = X[movie_idx]

    # Use NearestNeighbors to find similar movies
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(X)
    distances, indices = model.kneighbors(movie_vec, n_neighbors=k + 1)

    # Get neighbor movie IDs (skip the first one as it's the movie itself)
    neighbor_ids = [movie_inv_mapper[i] for i in indices.flatten()[1:]]
    recommendations = df[df['movieId'].isin(neighbor_ids)]['title'].unique()

    print(f"\nBecause you liked **{movie_title}**, you might also enjoy:")
    for i, rec in enumerate(recommendations[:k], 1):
        print(f"{i}. {rec}")
    
    return list(recommendations[:k])


def load_and_prepare_data(ratings_file='ratings.csv', movies_file=None):
    """
    Load and prepare the dataset for recommendation system.
    
    Parameters:
    ratings_file (str): Path to ratings CSV file
    movies_file (str): Optional path to movies CSV file for titles
    
    Returns:
    ratings: DataFrame with ratings data
    """
    try:
        ratings = pd.read_csv(ratings_file)
        print(f"Loaded {len(ratings)} ratings from {ratings_file}")
        print(f"Dataset shape: {ratings.shape}")
        print(f"\nFirst few rows:")
        print(ratings.head())
        
        # If movies file is provided and title column doesn't exist, merge to get titles
        if movies_file and 'title' not in ratings.columns:
            movies = pd.read_csv(movies_file)
            ratings = ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')
            print(f"\nMerged with movies dataset. Total unique movies: {ratings['movieId'].nunique()}")
        elif movies_file and 'title' in ratings.columns:
            print(f"\nTitle column already exists in ratings. Skipping merge.")
        
        return ratings
    except FileNotFoundError:
        print(f"Error: File '{ratings_file}' not found.")
        print("Please make sure the ratings.csv file is in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def display_statistics(df):
    """
    Display basic statistics about the dataset.
    
    Parameters:
    df (DataFrame): Ratings dataframe
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total ratings: {len(df)}")
    print(f"Unique users: {df['userId'].nunique()}")
    print(f"Unique movies: {df['movieId'].nunique()}")
    print(f"Average rating: {df['rating'].mean():.2f}")
    print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")
    print("\nRating distribution:")
    print(df['rating'].value_counts().sort_index())


def main():
    """
    Main function to run the recommendation system.
    """
    print("="*50)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("="*50)
    
    # Load data
    # Try loading with movies file first (if available)
    ratings = None
    import os
    if os.path.exists('movies.csv'):
        ratings = load_and_prepare_data('ratings.csv', 'movies.csv')
    else:
        ratings = load_and_prepare_data('ratings.csv')
    
    if ratings is None:
        print("\nPlease ensure you have a ratings.csv file.")
        print("The file should have columns: userId, movieId, rating")
        print("Optionally, you can also have movies.csv with columns: movieId, title")
        return
    
    # Check if 'title' column exists
    if 'title' not in ratings.columns:
        print("\nWarning: 'title' column not found in dataset.")
        print("Please ensure your dataset has a 'title' column or provide a movies.csv file.")
        return
    
    # Display statistics
    display_statistics(ratings)
    
    # Create user-item matrix
    print("\n" + "="*50)
    print("CREATING USER-ITEM MATRIX")
    print("="*50)
    X, movie_mapper, movie_inv_mapper = create_matrix(ratings)
    print(f"Matrix shape: {X.shape} (movies x users)")
    print(f"Matrix sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
    
    # Create pivot table for visualization
    user_item_matrix = ratings.pivot_table(
        index="title", columns="userId", values="rating"
    )
    print(f"\nPivot table shape: {user_item_matrix.shape}")
    print("\nFirst few rows and columns of user-item matrix:")
    print(user_item_matrix.iloc[:10, :5])
    
    # Example recommendations
    print("\n" + "="*50)
    print("GETTING RECOMMENDATIONS")
    print("="*50)
    
    # Get a sample movie title from the dataset
    sample_movies = ratings['title'].unique()[:5]
    print(f"\nSample movies in dataset: {list(sample_movies)}")
    
    # Try to recommend for a sample movie
    if len(sample_movies) > 0:
        test_movie = sample_movies[0]
        print(f"\nTesting recommendation for: {test_movie}")
        recommendations = recommend_similar(
            test_movie, ratings, X, movie_mapper, movie_inv_mapper, k=5
        )
    
    # Interactive mode
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Enter movie titles to get recommendations (or 'quit' to exit):")
    
    while True:
        movie_title = input("\nEnter a movie title: ").strip()
        if movie_title.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not movie_title:
            continue
        
        recommend_similar(
            movie_title, ratings, X, movie_mapper, movie_inv_mapper, k=5
        )


if __name__ == "__main__":
    main()

