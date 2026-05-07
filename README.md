# Netflix Recommendation System (NLP)

A content-based recommender that suggests Netflix titles similar to a query
title using **TF-IDF** vectorization and **cosine similarity** over a corpus
built from each title's plot, genres, cast, language and metadata.

The project ships as a reusable Python module (`recommender.py`), a CLI
(`app.py`) and an end-to-end Jupyter walkthrough (`final_function.ipynb`).

## Demo

```text
$ python app.py "Narcos" --top 5

Top 5 recommendations for 'Narcos':

 1. The Infiltrator (2016)  rating=7.0  score=0.280
     genres: Biography,Crime,Drama
 2. Un Bandido Honrado (2019)  rating=7.8  score=0.198
     genres: Comedy,Crime,Drama
 3. Cocaine Cowboys: Reloaded (2014)  rating=7.7  score=0.198
     genres: Crime,Documentary,History
 4. Suriname              rating=n/a  score=0.186
     genres: Biography,Crime,Drama
 5. Las muñecas de la mafia (2009)  rating=6.6  score=0.185
     genres: Drama
```

## Dataset

`netflix_list.csv` contains **7,008 Netflix titles** scraped from IMDb with
19 columns covering identifiers, ratings, plot, summary, cast, genres,
language, country and runtime. Title breakdown:

| Type            | Count |
| --------------- | ----: |
| Movies          | 2,923 |
| TV Series       | 2,199 |
| TV Episodes     |   785 |
| TV Specials     |   391 |
| TV Mini-Series  |   318 |
| TV Movies       |   161 |
| Other           |   231 |

## Methodology

1. **Cleaning** — drop identifier/image columns, strip whitespace, drop empty
   and duplicate titles, fill missing text fields with empty strings.
2. **Corpus construction** — concatenate `title`, `plot`, `summary`, `genres`,
   `cast`, `language` and `type` into a single lowercased document per title.
3. **Vectorization** — TF-IDF with unigrams + bigrams, English stop-words,
   `min_df=2`, sublinear TF and a 5,000-feature vocabulary cap.
4. **Similarity** — cosine similarity over the resulting sparse matrix,
   stored as `float32` to keep the full 7k × 7k matrix in memory.
5. **Ranking** — for a query title, return the top-N highest-scoring
   distinct titles along with their similarity score.

### Why TF-IDF + cosine?

* **Interpretable.** Every recommendation is explained by overlapping
  high-weight terms — useful for debugging and qualitative evaluation.
* **No cold-start.** Works on metadata only; no ratings or watch history
  required.
* **Cheap.** Fits in seconds on a laptop and serves recommendations in O(N)
  time per query.

## Installation

```bash
git clone https://github.com/Rushikesh-pawar/Netflix-Recomendation-System-NLP.git
cd Netflix-Recomendation-System-NLP

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Command line

```bash
# Top-5 recommendations for a title
python app.py "Stranger Things"

# Top-10 recommendations
python app.py "Stranger Things" --top 10

# Search the catalogue (handy for finding the exact title spelling)
python app.py --search "ozark"
```

### Python API

```python
from recommender import NetflixRecommender

rec = NetflixRecommender.from_csv("netflix_list.csv")

for r in rec.recommend("Narcos", top_n=5):
    print(f"{r.score:.3f}  {r.title} ({r.year})  {r.genres}")
```

### Notebook

Open `final_function.ipynb` for a step-by-step walkthrough covering loading,
EDA, preprocessing, vectorization, similarity and inference.

## Project structure

```
.
├── app.py                  # CLI entry point
├── recommender.py          # NetflixRecommender class (TF-IDF + cosine)
├── final_function.ipynb    # End-to-end Jupyter walkthrough
├── netflix_list.csv        # Dataset (7,008 titles, 19 columns)
├── requirements.txt
└── README.md
```

## Tech stack

* Python 3.10+
* pandas — data wrangling
* scikit-learn — `TfidfVectorizer`, `cosine_similarity`
* numpy — array ops

## Limitations & future work

* **Content-based only.** No collaborative signal (ratings, co-watch) — items
  with thin metadata get weaker recommendations.
* **Bag-of-words.** TF-IDF doesn't capture semantics; *"detective"* and
  *"investigator"* are unrelated terms. A sentence-embedding model
  (e.g. `sentence-transformers`) would lift quality at the cost of compute.
* **No personalization.** Same input → same output for every user.
* **Memory.** The full similarity matrix is dense (7k × 7k); for larger
  catalogues, switch to approximate nearest-neighbour (FAISS, HNSW).

Planned next steps:

- [ ] Swap TF-IDF for sentence-transformer embeddings and benchmark.
- [ ] Add a Streamlit / Gradio front-end with poster artwork.
- [ ] Hybrid model: combine content similarity with IMDb rating priors.
- [ ] Persist the fitted model (`joblib`) so cold-start time drops to ms.

## License

MIT
