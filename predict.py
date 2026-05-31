import joblib
import numpy as np


def load_model(model_path: str = "models/xgb_tuned.joblib"):
    """Load a trained model saved with joblib."""
    return joblib.load(model_path)


def predict_match(model, rank_diff: float, pts_diff: float, round_encoded: int, threshold: float = 0.5):
    """Predict whether Player_1 wins from engineered features.

    Args:
        model: Trained classifier loaded with joblib.
        rank_diff: Rank_1 - Rank_2.
        pts_diff: Pts_1 - Pts_2.
        round_encoded: Encoded round number (e.g. 1 for 1st Round, 7 for The Final).
        threshold: Probability threshold for classifying Player_1 win.

    Returns:
        dict with probability and predicted label.
    """
    X = np.array([[rank_diff, pts_diff, round_encoded]], dtype=float)
    proba = model.predict_proba(X)[:, 1][0]
    return {
        "prob_player1_win": float(proba),
        "predicted_label": int(proba >= threshold),
    }


def prepare_features(rank_1: float, rank_2: float, pts_1: float, pts_2: float, round_name: str):
    """Convert raw match values into model input features."""
    round_map = {
        "1st Round": 1,
        "2nd Round": 2,
        "3rd Round": 3,
        "4th Round": 4,
        "Quarterfinals": 5,
        "Semifinals": 6,
        "The Final": 7,
    }
    return {
        "rank_diff": rank_1 - rank_2,
        "pts_diff": pts_1 - pts_2,
        "round_encoded": round_map.get(round_name, 0),
    }


def predict_winner(model,rank_1,rank_2,pts_1,pts_2, round_name):
    features = prepare_features(rank_1,rank_2,pts_1,pts_2,round_name)

    result = predict_match(
        model,
        rank_diff=features["rank_diff"],
        pts_diff=features["pts_diff"],
        round_encoded=features["round_encoded"],
    )

    p1_win_prob = result["prob_player1_win"]

    winner = "Player_1" if p1_win_prob >= 0.5 else "Player_2"

    confidence = p1_win_prob if winner == "Player_1" else 1 - p1_win_prob

    return {
        "winner": winner,
        "confidence": round(confidence,2),
        "p1_win_prob": round(p1_win_prob,2),
    }


if __name__ == "__main__":
    model = load_model("models/xgb_tuned.joblib")

    winner = predict_winner(
        model,
        rank_1=1,
        rank_2=7,
        pts_1=8000,
        pts_2=10000,
        round_name="The Final"
    )

    print(winner)

