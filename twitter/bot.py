import random
from typing import Any, Dict, List, Optional


def generate_tweet(prediction: Dict[str, Any]) -> str:
    """Generate a creative tweet from a game prediction dictionary."""
    home = prediction["home_team"]
    away = prediction["away_team"]
    h_pred = round(float(prediction["home_pred"]))
    a_pred = round(float(prediction["away_pred"]))
    winner = prediction["predicted_winner"]
    loser = away if winner == home else home
    margin = abs(float(prediction["margin"]))
    conf = str(prediction.get("confidence", "MEDIUM")).upper()
    is_playoff = bool(prediction.get("is_playoff", False))

    # Top scorer from each team
    home_preds = prediction.get("home_player_preds", [])
    away_preds = prediction.get("away_player_preds", [])

    home_star = max(home_preds, key=lambda x: x.get("pred_points", 0)) if home_preds else None
    away_star = max(away_preds, key=lambda x: x.get("pred_points", 0)) if away_preds else None
    winner_star = home_star if winner == home else away_star

    # Series context
    series_ctx = ""
    for factor in prediction.get("factors", []):
        if "Series" in factor or "Game" in factor:
            series_ctx = factor
            break

    game_tag = "🏆" if is_playoff else "🏀"
    conf_line = {
        "HIGH": "i'm calling it early —",
        "MEDIUM": "the numbers say —",
        "LOW": "this one's a coin flip but —",
    }.get(conf, "model says —")

    # Player highlight
    star_line = ""
    if winner_star:
        flags = ""
        if winner_star.get("hot_flag"):
            flags = " 🔥"
        elif winner_star.get("slump_flag"):
            flags = " 📉"

        star_line = (
            f"\n{winner_star['full_name']} goes "
            f"{round(float(winner_star.get('pred_points', 0)))}pts{flags}"
        )

    hashtag = "#NBAPlayoffs" if is_playoff else "#NBA"

    templates = [
        (
            f"{game_tag} {away} @ {home}\n"
            f"{conf_line} {winner} wins\n"
            f"predicted: {home} {h_pred} — {away} {a_pred}"
            f"{star_line}\n{hashtag}"
        ),
        (
            f"ran the model. {loser} fans look away 👀\n"
            f"{game_tag} {winner} {max(h_pred, a_pred)} — "
            f"{loser} {min(h_pred, a_pred)}"
            f"{star_line}\n{hashtag}"
        ),
        (
            f"{game_tag} {away} @ {home} prediction\n"
            f"{conf_line} {winner} by {margin:.0f}\n"
            f"final: {home} {h_pred} — {away} {a_pred}"
            f"{star_line}\n{hashtag}"
        ),
        (
            f"good morning to everyone except {loser} fans 🙂\n"
            f"{winner} wins tonight — {home} {h_pred} {away} {a_pred}"
            f"{star_line}\n{hashtag}"
        ),
    ]

    # Add playoff series context if available
    if series_ctx and is_playoff:
        templates.append(
            (
                f"{game_tag} {series_ctx}\n"
                f"{conf_line} {winner} wins\n"
                f"{home} {h_pred} — {away} {a_pred}"
                f"{star_line}\n{hashtag}"
            )
        )

    tweet = random.choice(templates)

    # Keep under 280 chars
    if len(tweet) > 275:
        tweet = tweet[:272] + "..."

    return tweet


def generate_tweets(predictions: List[Dict[str, Any]]) -> List[str]:
    """Generate tweets for a list of predictions."""
    tweets = []
    for prediction in predictions:
        try:
            tweets.append(generate_tweet(prediction))
        except KeyError as exc:
            tweets.append(f"Skipping prediction due to missing key: {exc}")
    return tweets


def example_predictions() -> List[Dict[str, Any]]:
    """Return sample predictions for testing."""
    return [
        {
            "home_team": "Lakers",
            "away_team": "Nuggets",
            "home_pred": 108.4,
            "away_pred": 114.9,
            "predicted_winner": "Nuggets",
            "margin": 6.5,
            "confidence": "HIGH",
            "is_playoff": True,
            "factors": ["Series tied 1-1", "Game 3 in Los Angeles"],
            "home_player_preds": [
                {"full_name": "LeBron James", "pred_points": 27.2, "hot_flag": False, "slump_flag": False},
                {"full_name": "Anthony Davis", "pred_points": 24.8, "hot_flag": False, "slump_flag": False},
            ],
            "away_player_preds": [
                {"full_name": "Nikola Jokic", "pred_points": 29.4, "hot_flag": True, "slump_flag": False},
                {"full_name": "Jamal Murray", "pred_points": 22.1, "hot_flag": False, "slump_flag": False},
            ],
        },
        {
            "home_team": "Celtics",
            "away_team": "Heat",
            "home_pred": 112.1,
            "away_pred": 104.7,
            "predicted_winner": "Celtics",
            "margin": 7.4,
            "confidence": "MEDIUM",
            "is_playoff": False,
            "factors": ["Boston rest advantage"],
            "home_player_preds": [
                {"full_name": "Jayson Tatum", "pred_points": 28.6, "hot_flag": True, "slump_flag": False},
                {"full_name": "Jaylen Brown", "pred_points": 24.5, "hot_flag": False, "slump_flag": False},
            ],
            "away_player_preds": [
                {"full_name": "Jimmy Butler", "pred_points": 23.3, "hot_flag": False, "slump_flag": False},
                {"full_name": "Bam Adebayo", "pred_points": 19.2, "hot_flag": False, "slump_flag": False},
            ],
        },
    ]


def main() -> None:
    """Run a simple demo."""
    predictions = example_predictions()
    tweets = generate_tweets(predictions)

    for i, tweet in enumerate(tweets, start=1):
        print(f"\n--- Tweet {i} ---")
        print(tweet)
        print(f"\nLength: {len(tweet)} characters")


if __name__ == "__main__":
    main()