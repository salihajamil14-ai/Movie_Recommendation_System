# Movie Recommendation System - Submission Package

## 📁 Files Included

1. **`movie_recommendation_system.py`** - Main recommendation system code (SUBMISSION FILE)
2. **`ratings.csv`** - Movie ratings dataset (100,836 ratings)
3. **`movies.csv`** - Movie information dataset (9,724 movies)
4. **`requirements.txt`** - Python package dependencies
5. **`README.md`** - Detailed documentation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the System
```bash
python movie_recommendation_system.py
```

## 📋 Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## 🎯 Features

- **Content-Based Filtering**: Uses cosine similarity to find similar movies
- **Sparse Matrix Implementation**: Efficient handling of large datasets
- **Interactive Mode**: Get recommendations by entering movie titles
- **Comprehensive Statistics**: Displays dataset information and matrix details
- **Error Handling**: Robust error handling for missing data or invalid inputs

## 📊 Dataset Information

- **Source**: MovieLens Latest Small Dataset
- **Total Ratings**: 100,836
- **Users**: 610
- **Movies**: 9,724
- **Rating Scale**: 0.5 - 5.0 (in increments of 0.5)

## 🔧 How It Works

1. **Data Loading**: Loads ratings and movies from CSV files
2. **Matrix Creation**: Creates a sparse user-item matrix (movies × users)
3. **Similarity Calculation**: Uses NearestNeighbors with cosine similarity
4. **Recommendation**: Returns k most similar movies based on rating patterns

## 💡 Example Usage

```
Enter a movie title: Toy Story (1995)

Because you liked: Toy Story (1995)
You might also enjoy:
   1. Star Wars: Episode IV - A New Hope (1977)
   2. Forrest Gump (1994)
   3. Jurassic Park (1993)
   4. Independence Day (a.k.a. ID4) (1996)
   5. Toy Story 2 (1999)
```

## 📝 Code Structure

- `create_matrix()` - Creates sparse user-item matrix
- `recommend_similar()` - Generates movie recommendations
- `load_and_prepare_data()` - Loads and prepares dataset
- `display_statistics()` - Shows dataset statistics
- `main()` - Main execution function

## 🎓 Algorithm

The system uses **Collaborative Filtering** with **Cosine Similarity**:
- Creates a user-item matrix where rows = movies, columns = users
- Uses cosine similarity to find movies with similar rating patterns
- Recommends movies that users with similar tastes have rated highly

## 📚 References

Based on: https://www.geeksforgeeks.org/machine-learning/recommendation-system-in-python/

Dataset: MovieLens Latest Small Dataset (https://grouplens.org/datasets/movielens/)

## ✅ Testing

The code has been tested and verified to work with:
- MovieLens dataset (100K+ ratings)
- Sample generated data
- Various movie titles and edge cases

---

**Note**: The main submission file is `movie_recommendation_system.py`

