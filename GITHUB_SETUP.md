# GitHub Setup Instructions

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name your repository (e.g., "movie-recommendation-system")
5. Choose Public or Private
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these commands:

```bash
# Add the remote repository (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Alternative - Using SSH (if you have SSH keys set up)

```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## What's Included in This Repository

- `movie_recommendation_system.py` - Main recommendation system (submission file)
- `recommendation_system.py` - Alternative version
- `download_dataset.py` - Script to download MovieLens dataset
- `generate_sample_data.py` - Script to generate sample data
- `requirements.txt` - Python dependencies
- `README.md` - Documentation
- `.gitignore` - Excludes large dataset files

## Note About Dataset Files

The dataset files (`ratings.csv`, `movies.csv`, `ml-latest-small/`) are excluded from git because they are large. Users can:
1. Run `python download_dataset.py` to download the dataset
2. Or use `python generate_sample_data.py` to generate sample data

## Troubleshooting

If you get authentication errors:
- Use GitHub Personal Access Token instead of password
- Or set up SSH keys for easier authentication

If you need to update the remote URL:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

