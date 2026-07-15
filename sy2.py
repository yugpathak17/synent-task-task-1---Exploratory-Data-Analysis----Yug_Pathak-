from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATASET_PATH = Path(r"C:\Users\YUG M PATHAK\Downloads\netflix_dataset.csv")
OUTPUT_DIR = Path(__file__).resolve().parent

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nSummary Statistics:")
print(df.describe(include='all').T)

for col in df.columns:
    if col != "title":
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            pass

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

print("\nMissing values:\n", df.isnull().sum())

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    print("\nCorrelation Matrix:\n", corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png')
    plt.show()

if 'release_year' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df['release_year'], bins=20, kde=True)
    plt.title('Distribution of Release Year')
    plt.xlabel('Release Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'release_year_distribution.png')
    plt.show()

if 'genre' in df.columns:
    top_genres = df['genre'].astype(str).value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
    plt.title('Top Genres')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_genres.png')
    plt.show()

if 'rating' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=df['rating'], order=df['rating'].value_counts().index)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rating_distribution.png')
    plt.show()

print("\nInsights:")
if 'release_year' in df.columns:
    print(f"- Most titles were released around {df['release_year'].mode().iloc[0]}.")
if 'rating' in df.columns:
    print(f"- Most common rating is: {df['rating'].mode().iloc[0]}.")
if len(numeric_cols) > 1:
    corr_abs = corr.abs().copy()
    corr_abs = corr_abs.where(~np.eye(len(corr_abs), dtype=bool), np.nan)
    strongest_corr = corr_abs.unstack().dropna().sort_values(ascending=False)
    top_corr = strongest_corr.head(3)
    print("- Strongest numeric relationships:")
    for (col1, col2), value in top_corr.items():
        print(f"  * {col1} vs {col2}: {value:.2f}")
