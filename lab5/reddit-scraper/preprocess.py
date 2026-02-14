import re
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from database import get_db


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_TEXT_RE = re.compile(r"[^a-z0-9\s]+")


TOPIC_RULES = [
    ("ransomware", ["ransomware", "decrypt", "extortion"]),
    ("phishing", ["phishing", "phish", "spoof", "credential", "oauth"]),
    ("malware", ["malware", "trojan", "worm", "payload", "botnet", "virus"]),
    ("vulnerability", ["cve", "vuln", "vulnerability", "exploit", "patch"]),
    ("network_security", ["firewall", "ids", "ips", "siem", "soc", "zero trust"]),
    ("privacy", ["privacy", "gdpr", "pii", "anonym", "tracking"]),
    ("forensics", ["forensic", "incident response", "ir", "triage", "log analysis"]),
]


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.lower()
    text = NON_TEXT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_topic(text: str) -> str:
    if not text:
        return "unknown"
    for topic, keywords in TOPIC_RULES:
        for kw in keywords:
            if kw in text:
                return topic
    return "general"


def simple_keywords(text: str, top_k: int = 10) -> str:
    """
    Minimal keyword extractor:
    - removes very short tokens
    - counts token frequency
    - returns top_k tokens joined by commas
    This is a baseline; later we can replace with TF-IDF/TextRank.
    """
    if not text:
        return ""
    tokens = [t for t in text.split() if len(t) >= 4]
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return ",".join([w for w, _ in top])


def build_cleaned_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    post_id = raw["post_id"]
    subreddit = raw["subreddit"]

    title = raw.get("title") or ""
    body = raw.get("selftext") or ""
    flair = raw.get("flair") or ""
    author = raw.get("author") or ""

    title_clean = normalize_text(title)
    body_clean = normalize_text(body)

    combined = (title_clean + " " + body_clean).strip()

    author_anon = hashlib.sha256(author.encode("utf-8", errors="ignore")).hexdigest()
    created_utc = raw.get("created_utc")
    created_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc).replace(tzinfo=None) if created_utc else None

    is_ad = "[AD]" in flair

    keywords = simple_keywords(combined, top_k=10)
    topic = infer_topic(combined)

    cleaned = {
        "post_id": post_id,
        "subreddit": subreddit,
        "title_clean": title_clean,
        "body_clean": body_clean,
        "author_anon": author_anon,
        "score": int(raw.get("score") or 0),
        "num_comments": int(raw.get("num_comments") or 0),
        "created_dt": created_dt,
        "url": raw.get("url") or "",
        "is_ad": bool(is_ad),
        "ocr_text": "",
        "keywords": keywords,
        "topic": topic,
    }
    return cleaned


def main(batch_size: int = 300, max_batches: Optional[int] = None) -> None:
    db = get_db()

    total_inserted = 0
    batch_count = 0

    while True:
        if max_batches is not None and batch_count >= max_batches:
            break

        raw_posts = db.get_unprocessed_posts(limit=batch_size)
        if not raw_posts:
            break

        cleaned_batch: List[Dict[str, Any]] = []
        for r in raw_posts:
            cleaned_batch.append(build_cleaned_record(r))

        affected = db.insert_cleaned_posts_batch(cleaned_batch)
        total_inserted += affected
        batch_count += 1

        print(f"Processed batch {batch_count}: raw={len(raw_posts)} inserted_or_updated={affected} total={total_inserted}")

    print("Preprocessing done.")
    print(f"Total inserted_or_updated: {total_inserted}")


if __name__ == "__main__":
    main()
