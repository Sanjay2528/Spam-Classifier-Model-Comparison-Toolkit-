"""
generate_sample_data.py
------------------------
Generates a small, self-contained labelled dataset (ham/spam messages)
so the project runs end-to-end without requiring an external download.
For a real submission you may replace data/sample_dataset.csv with a
larger public dataset (e.g. the UCI SMS Spam Collection) - just keep the
same two columns: label, message.
"""

import csv
import os
import random

random.seed(42)

SPAM_TEMPLATES = [
    "Congratulations! You have WON a free {prize}. Call {phone} now to claim!",
    "URGENT: Your account will be suspended. Verify now at {url}",
    "You've been selected for a FREE {prize}! Text YES to {phone}",
    "Limited time offer: Get {prize} for $0 today only. Click {url}",
    "WINNER!! As a valued customer you have been selected to receive a {prize}",
    "Claim your prize now! You won a {prize} in our lucky draw. Reply WIN",
    "Cheap loans available instantly, no credit check. Apply at {url}",
    "Your loan of $5000 has been approved. Confirm details at {url}",
    "FREE entry into our $1000 weekly draw, just text ENTER to {phone}",
    "Hot singles in your area want to chat! Visit {url} now",
]

HAM_TEMPLATES = [
    "Hey, are we still meeting for lunch at {time}?",
    "Can you send me the notes from today's {subject} class?",
    "Don't forget to submit the {subject} assignment by {time}.",
    "Happy birthday! Hope you have a great day.",
    "I'll be home around {time}, see you then.",
    "Thanks for your help with the {subject} project yesterday.",
    "Let's catch up this weekend, are you free?",
    "The meeting has been moved to {time}, please update your calendar.",
    "Can you pick up some groceries on your way home?",
    "Great job on the presentation today!",
]

PRIZES = ["iPhone", "cash prize", "holiday voucher", "gift card", "laptop"]
PHONES = ["09012345678", "08001234567", "07123456789"]
URLS = ["http://claim-prize.example.com", "http://verify-account.example.net"]
TIMES = ["6pm", "noon", "5:30", "tomorrow morning"]
SUBJECTS = ["AI/ML", "math", "history", "chemistry"]


def make_dataset(n_per_class: int = 120):
    rows = []
    for _ in range(n_per_class):
        t = random.choice(SPAM_TEMPLATES)
        msg = t.format(
            prize=random.choice(PRIZES),
            phone=random.choice(PHONES),
            url=random.choice(URLS),
        )
        rows.append(("spam", msg))

    for _ in range(n_per_class):
        t = random.choice(HAM_TEMPLATES)
        msg = t.format(time=random.choice(TIMES), subject=random.choice(SUBJECTS))
        rows.append(("ham", msg))

    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_dataset.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = make_dataset()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "message"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
