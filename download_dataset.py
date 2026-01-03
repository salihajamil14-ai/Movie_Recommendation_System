"""
Script to download and prepare MovieLens dataset for the recommendation system.
"""

import zipfile
import os
import pandas as pd
import urllib.request

def download_and_prepare_movielens():
    """Download and prepare MovieLens dataset."""
    
    # Check if already downloaded
    if not os.path.exists('ml-latest-small.zip'):
        print("Downloading MovieLens dataset...")
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        urllib.request.urlretrieve(url, 'ml-latest-small.zip')
        print("Download complete!")
    else:
        print("MovieLens zip file already exists.")
    
    # Extract if not already extracted
    if not os.path.exists('ml-latest-small'):
        print("\nExtracting dataset...")
        with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Extraction complete!")
    else:
        print("Dataset already extracted.")
    
    # Load and prepare the files
    print("\nPreparing data files...")
    
    # Load ratings
    ratings_path = 'ml-latest-small/ratings.csv'
    ratings = pd.read_csv(ratings_path)
    print(f"Loaded {len(ratings)} ratings")
    
    # Load movies
    movies_path = 'ml-latest-small/movies.csv'
    movies = pd.read_csv(movies_path)
    print(f"Loaded {len(movies)} movies")
    
    # Merge ratings with movies to get titles
    ratings_with_titles = ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')
    
    # Reorder columns to match expected format
    ratings_with_titles = ratings_with_titles[['userId', 'movieId', 'title', 'rating']]
    
    # Save as ratings.csv (overwrite existing)
    ratings_with_titles.to_csv('ratings.csv', index=False)
    print(f"\nSaved ratings.csv with {len(ratings_with_titles)} ratings")
    
    # Save movies.csv
    movies.to_csv('movies.csv', index=False)
    print(f"Saved movies.csv with {len(movies)} movies")
    
    # Display statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total ratings: {len(ratings_with_titles):,}")
    print(f"Unique users: {ratings_with_titles['userId'].nunique():,}")
    print(f"Unique movies: {ratings_with_titles['movieId'].nunique():,}")
    print(f"Average rating: {ratings_with_titles['rating'].mean():.2f}")
    print(f"Rating range: {ratings_with_titles['rating'].min()} - {ratings_with_titles['rating'].max()}")
    
    print("\nFirst few rows of ratings.csv:")
    print(ratings_with_titles.head(10))
    
    print("\n" + "="*50)
    print("Dataset ready! You can now run recommendation_system.py")
    print("="*50)

if __name__ == "__main__":
    download_and_prepare_movielens()

