"""Content-based Netflix recommender using TF-IDF and cosine similarity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_TEXT_COLUMNS: tuple[str, ...] = (
    "title",
    "plot",
    "summary",
    "genres",
    "cast",
    "language",
    "type",
)

DROP_COLUMNS: tuple[str, ...] = ("imdb_id", "certificate", "image_url")

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Recommendation:
    title: str
    score: float
    genres: str
    rating: float | None
    year: int | None


class NetflixRecommender:
    """Content-based recommender over the Netflix titles dataset.

    Combines several text columns into a single corpus, vectorizes with
    TF-IDF, and ranks candidates by cosine similarity to a query title.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        text_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS,
    ) -> None:
        self.text_columns = tuple(text_columns)
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
            sublinear_tf=True,
        )
        self.df: pd.DataFrame | None = None
        self.similarity_: np.ndarray | None = None
        self._title_to_index: dict[str, int] = {}

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "NetflixRecommender":
        recommender = cls(**kwargs)
        recommender.fit(pd.read_csv(path))
        return recommender

    def fit(self, df: pd.DataFrame) -> "NetflixRecommender":
        self.df = self._prepare(df)
        corpus = self._build_corpus(self.df)
        vectors = self.vectorizer.fit_transform(corpus)
        self.similarity_ = cosine_similarity(vectors).astype(np.float32)
        self._title_to_index = {
            title.casefold(): idx
            for idx, title in enumerate(self.df["title"].tolist())
        }
        return self

    def recommend(self, title: str, top_n: int = 5) -> list[Recommendation]:
        if self.df is None or self.similarity_ is None:
            raise RuntimeError("Call fit() before recommend().")

        index = self._lookup_index(title)
        scores = self.similarity_[index]
        ranked = np.argsort(-scores)
        results: list[Recommendation] = []
        for candidate in ranked:
            if candidate == index:
                continue
            row = self.df.iloc[candidate]
            results.append(
                Recommendation(
                    title=row["title"],
                    score=float(scores[candidate]),
                    genres=str(row.get("genres", "") or ""),
                    rating=_safe_float(row.get("rating")),
                    year=_safe_int(row.get("startYear")),
                )
            )
            if len(results) >= top_n:
                break
        return results

    def search_titles(self, query: str, limit: int = 10) -> list[str]:
        if self.df is None:
            raise RuntimeError("Call fit() before search_titles().")
        needle = query.casefold()
        matches = [t for t in self.df["title"] if needle in t.casefold()]
        return matches[:limit]

    def _lookup_index(self, title: str) -> int:
        key = title.casefold()
        if key in self._title_to_index:
            return self._title_to_index[key]
        suggestions = self.search_titles(title, limit=5)
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        raise KeyError(f"Title {title!r} not found in the dataset.{hint}")

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
        df = df.dropna(subset=["title"]).reset_index(drop=True)
        df["title"] = df["title"].astype(str).str.strip()
        df = df[df["title"].str.len() > 0]
        df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
        for column in self.text_columns:
            if column in df.columns:
                df[column] = df[column].fillna("").astype(str).str.strip()
        return df

    def _build_corpus(self, df: pd.DataFrame) -> list[str]:
        available = [c for c in self.text_columns if c in df.columns]
        joined = df[available].agg(" ".join, axis=1)
        return [_normalize(text) for text in joined]


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(result) else result


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
