# Movie Recommendation System

A content-based movie recommendation system built in Python using collaborative filtering and cosine similarity. This system recommends movies similar to ones a user has liked, based on user ratings patterns from the MovieLens dataset.

A content-based movie recommendation system built in Python using collaborative filtering and cosine similarity. This system recommends movies similar to ones a user has liked, based on user ratings patterns.

## Features

- **Content-Based Filtering**: Recommends movies based on similarity to movies you've liked
- **Cosine Similarity**: Uses cosine similarity to find similar movies in the user-item matrix
- **Interactive Mode**: Get recommendations by entering movie titles
- **Data Visualization**: Displays dataset statistics and matrix information

## Requirements

- Python 3.7+
- Required packages (see `requirements.txt`)

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The system expects a CSV file named `ratings.csv` with the following columns:
- `userId`: Unique identifier for each user
- `movieId`: Unique identifier for each movie
- `rating`: Rating given by user (typically 1-5)
- `title`: Movie title (optional, can be merged from separate movies.csv)

Optionally, you can also provide `movies.csv` with:
- `movieId`: Unique identifier for each movie
- `title`: Movie title

### Using Sample Data

If you don't have a dataset, you can generate sample data:

```bash
python generate_sample_data.py
```

This will create:
- `ratings.csv`: Sample ratings data
- `movies.csv`: Movie titles and IDs

### Using Real Data

You can download real movie ratings datasets from:
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- Extract the files and ensure they're named `ratings.csv` and `movies.csv`

## Usage

### Basic Usage

Run the recommendation system:

```bash
python recommendation_system.py
```

The script will:
1. Load the dataset
2. Display statistics
3. Create the user-item matrix
4. Show sample recommendations
5. Enter interactive mode where you can input movie titles

### Interactive Mode

Once the system is running, you can enter movie titles to get recommendations:

```
Enter a movie title: The Dark Knight

Because you liked **The Dark Knight**, you might also enjoy:
1. Batman Begins
2. Inception
3. Interstellar
4. The Prestige
5. Memento
```

Type `quit`, `exit`, or `q` to exit.

## How It Works

1. **Data Loading**: Loads ratings data from CSV files
2. **Matrix Creation**: Creates a sparse user-item matrix where:
   - Rows = Movies
   - Columns = Users
   - Values = Ratings
3. **Similarity Calculation**: Uses NearestNeighbors with cosine similarity to find movies with similar rating patterns
4. **Recommendation**: Returns the k most similar movies based on the input movie

## Code Structure

- `recommendation_system.py`: Main recommendation system implementation
- `generate_sample_data.py`: Script to generate sample data for testing
- `requirements.txt`: Python package dependencies

## Functions

### `create_matrix(df)`
Creates a sparse user-item matrix from ratings data.

### `recommend_similar(movie_title, df, X, movie_mapper, movie_inv_mapper, k=5)`
Recommends k similar movies based on a given movie title.

### `load_and_prepare_data(ratings_file, movies_file=None)`
Loads and prepares the dataset for the recommendation system.

## Example Output

```
==================================================
MOVIE RECOMMENDATION SYSTEM
==================================================
Loaded 2000 ratings from ratings.csv
Dataset shape: (2000, 4)

First few rows:
   userId  movieId           title  rating
0       1        1  The Dark Knight     5
1       1        2        Inception     4
...

==================================================
DATASET STATISTICS
==================================================
Total ratings: 2000
Unique users: 100
Unique movies: 50
Average rating: 3.85
Rating range: 1 - 5
```

## Limitations

- Requires sufficient ratings data for accurate recommendations
- Performance may degrade with very large datasets (millions of ratings)
- Cold start problem: New movies with few ratings may not be recommended well

## Future Improvements

- Implement hybrid recommendation (combining content-based and collaborative filtering)
- Add user-based recommendations
- Implement matrix factorization techniques
- Add web interface
- Optimize for larger datasets

## References

Based on the tutorial from [GeeksforGeeks - Recommendation System in Python](https://www.geeksforgeeks.org/machine-learning/recommendation-system-in-python/)

## License

This project is open source and available for educational purposes.

