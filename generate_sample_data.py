"""
Script to generate sample movie ratings data for testing the recommendation system.
This creates a synthetic dataset similar to MovieLens format.
"""

import pandas as pd
import numpy as np
import random

def generate_sample_data(num_users=100, num_movies=50, num_ratings=1000, output_file='ratings.csv'):
    """
    Generate sample movie ratings data.
    
    Parameters:
    num_users (int): Number of users
    num_movies (int): Number of movies
    num_ratings (int): Number of ratings to generate
    output_file (str): Output CSV filename
    """
    # Movie titles (sample)
    movie_titles = [
        "The Dark Knight", "Inception", "Interstellar", "The Matrix", "Pulp Fiction",
        "Fight Club", "Forrest Gump", "The Shawshank Redemption", "The Godfather",
        "Titanic", "Avatar", "Jurassic Park", "Star Wars", "The Avengers", "Iron Man",
        "Spider-Man", "Batman Begins", "The Prestige", "Memento", "Dunkirk",
        "The Dark Knight Rises", "Interstellar", "The Departed", "Goodfellas", "Casino",
        "Scarface", "Heat", "Gladiator", "Braveheart", "Saving Private Ryan",
        "Schindler's List", "The Green Mile", "The Sixth Sense", "Shutter Island",
        "Inception", "The Revenant", "The Wolf of Wall Street", "Django Unchained",
        "Inglourious Basterds", "Kill Bill", "Reservoir Dogs", "The Hateful Eight",
        "Once Upon a Time in Hollywood", "Joker", "Parasite", "1917", "Dune",
        "Blade Runner 2049", "Mad Max: Fury Road", "The Grand Budapest Hotel"
    ]
    
    # Ensure we have enough unique titles
    if num_movies > len(movie_titles):
        # Generate additional titles
        for i in range(len(movie_titles), num_movies):
            movie_titles.append(f"Movie {i+1}")
    
    # Create movies dataframe
    movies_data = {
        'movieId': range(1, num_movies + 1),
        'title': movie_titles[:num_movies]
    }
    movies_df = pd.DataFrame(movies_data)
    movies_df.to_csv('movies.csv', index=False)
    print(f"Created movies.csv with {num_movies} movies")
    
    # Generate ratings
    ratings_data = []
    user_ids = list(range(1, num_users + 1))
    movie_ids = list(range(1, num_movies + 1))
    
    # Create some patterns: users who like similar movies
    for _ in range(num_ratings):
        user_id = random.choice(user_ids)
        movie_id = random.choice(movie_ids)
        
        # Simulate rating (1-5 scale, with some bias)
        # Users tend to rate movies they like higher
        base_rating = random.choices(
            [1, 2, 3, 4, 5],
            weights=[0.1, 0.15, 0.2, 0.3, 0.25]  # More 4s and 5s
        )[0]
        
        ratings_data.append({
            'userId': user_id,
            'movieId': movie_id,
            'rating': base_rating
        })
    
    ratings_df = pd.DataFrame(ratings_data)
    
    # Remove duplicates (same user rating same movie twice)
    ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'])
    
    # Merge with movies to get titles
    ratings_df = ratings_df.merge(movies_df[['movieId', 'title']], on='movieId', how='left')
    
    # Reorder columns
    ratings_df = ratings_df[['userId', 'movieId', 'title', 'rating']]
    
    # Save to CSV
    ratings_df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(ratings_df)} ratings")
    print(f"Unique users: {ratings_df['userId'].nunique()}")
    print(f"Unique movies: {ratings_df['movieId'].nunique()}")
    
    return ratings_df, movies_df


if __name__ == "__main__":
    print("Generating sample movie ratings data...")
    print("="*50)
    
    # Generate sample data
    ratings_df, movies_df = generate_sample_data(
        num_users=100,
        num_movies=50,
        num_ratings=2000
    )
    
    print("\n" + "="*50)
    print("Sample data preview:")
    print(ratings_df.head(10))
    print("\nYou can now run recommendation_system.py with this data!")

