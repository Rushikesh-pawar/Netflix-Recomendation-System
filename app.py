"""CLI for the Netflix content-based recommender.

Examples
--------
    python app.py "Narcos"
    python app.py "Stranger Things" --top 10
    python app.py --search "ozark"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recommender import NetflixRecommender

DATA_PATH = Path(__file__).resolve().parent / "netflix_list.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend Netflix titles similar to a given title."
    )
    parser.add_argument("title", nargs="?", help="Title to base recommendations on.")
    parser.add_argument("--top", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--search",
        metavar="QUERY",
        help="Print titles whose name contains QUERY and exit.",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_PATH),
        help=f"Path to the dataset CSV (default: {DATA_PATH.name}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recommender = NetflixRecommender.from_csv(args.data)

    if args.search:
        for hit in recommender.search_titles(args.search, limit=20):
            print(hit)
        return 0

    if not args.title:
        print("Error: provide a title or use --search QUERY.", file=sys.stderr)
        return 2

    try:
        recs = recommender.recommend(args.title, top_n=args.top)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Top {len(recs)} recommendations for {args.title!r}:\n")
    for i, rec in enumerate(recs, start=1):
        rating = f"{rec.rating:.1f}" if rec.rating is not None else "n/a"
        year = rec.year if rec.year is not None else "----"
        print(f"{i:>2}. {rec.title} ({year})  rating={rating}  score={rec.score:.3f}")
        if rec.genres:
            print(f"     genres: {rec.genres}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
