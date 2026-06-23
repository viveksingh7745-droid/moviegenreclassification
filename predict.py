"""
predict.py
Classifies movie genres from user-supplied titles and overviews.
Usage:
    python predict.py                    ← runs built-in demo predictions
    python predict.py --interactive      ← type your own movie descriptions
"""

import pickle
import re
import sys

MODEL_FILE   = "models/genre_classifier.pkl"
ENCODER_FILE = "models/label_encoder.pkl"

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","it","its","he","she","they","we","you","i","this",
    "that","his","her","their","be","as","are","was","were","been","have",
    "has","had","who","which","when","where","while","after","before","into",
    "up","out","about","after","over","into","then","than","so","do","does",
    "did","not","no","but","if","all","each","both","more","also","just",
    "only","even","what","there","him","them","us","will","would","could",
    "should","may","might","can","an","am","through","during","between",
}

GENRE_EMOJI = {
    "Action":    "💥",
    "Comedy":    "😂",
    "Drama":     "🎭",
    "Horror":    "👻",
    "Romance":   "❤️",
    "Sci-Fi":    "🚀",
    "Thriller":  "🔪",
    "Animation": "🎨",
}

DEMO_MOVIES = [
    {
        "title":    "The Avengers",
        "overview": "Earth's mightiest heroes must come together and learn to fight as a team to stop the mischievous Loki from enslaving humanity with the help of an alien army.",
    },
    {
        "title":    "Crazy, Stupid, Love",
        "overview": "A middle-aged husband's life changes when his wife asks for a divorce and a smooth player teaches him how to become a real man again.",
    },
    {
        "title":    "Alien",
        "overview": "The crew of a commercial spacecraft encounters a deadly extraterrestrial creature after investigating an uncharted planet in deep space.",
    },
    {
        "title":    "The Notebook",
        "overview": "A poor but passionate young man falls in love with a rich girl and they are separated by her snobbish parents.",
    },
    {
        "title":    "Parasite",
        "overview": "Greed and class discrimination threaten the symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
    },
    {
        "title":    "Toy Story",
        "overview": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
    },
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def load_model():
    with open(MODEL_FILE,   "rb") as f: pipeline = pickle.load(f)
    with open(ENCODER_FILE, "rb") as f: le       = pickle.load(f)
    return pipeline, le


def predict_genre(pipeline, le, title: str, overview: str):
    """Return predicted genre + confidence scores for all genres."""
    text  = clean_text(title + " " + overview)
    probs = pipeline.predict_proba([text])[0]
    idx   = probs.argmax()
    genres = le.classes_

    predicted  = genres[idx]
    confidence = probs[idx] * 100
    top3 = sorted(zip(genres, probs), key=lambda x: -x[1])[:3]
    return predicted, confidence, top3


def print_result(title, overview, pipeline, le):
    predicted, confidence, top3 = predict_genre(pipeline, le, title, overview)
    emoji = GENRE_EMOJI.get(predicted, "🎬")
    print(f"\n  Movie   : {title}")
    print(f"  Overview: {overview[:80]}{'...' if len(overview) > 80 else ''}")
    print(f"  ──────────────────────────────────────────")
    print(f"  Prediction : {emoji}  {predicted}  ({confidence:.1f}% confidence)")
    print(f"  Top 3      :", "  |  ".join(
        f"{GENRE_EMOJI.get(g,'🎬')} {g} {p*100:.1f}%" for g, p in top3
    ))


def run_demo(pipeline, le):
    print("\n" + "=" * 60)
    print("  MOVIE GENRE CLASSIFIER — Demo Predictions")
    print("=" * 60)
    for movie in DEMO_MOVIES:
        print_result(movie["title"], movie["overview"], pipeline, le)
    print("\n" + "=" * 60)


def run_interactive(pipeline, le):
    print("\n" + "=" * 60)
    print("  MOVIE GENRE CLASSIFIER — Interactive Mode")
    print("  Type 'quit' to exit")
    print("=" * 60)
    while True:
        print()
        title    = input("  Movie title   : ").strip()
        if title.lower() in ("quit", "exit", "q"):
            break
        overview = input("  Movie overview: ").strip()
        if not overview:
            print("  Please enter an overview.")
            continue
        print_result(title, overview, pipeline, le)


def main():
    pipeline, le = load_model()
    if "--interactive" in sys.argv:
        run_interactive(pipeline, le)
    else:
        run_demo(pipeline, le)
        print("\nTip: Run  python predict.py --interactive  to classify your own movies!")


if __name__ == "__main__":
    main()
