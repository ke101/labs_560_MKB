import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "DSCI560-Lab2/1.0 (academic; contact: yi-hsien@usc.edu)"
}

def clean_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def scrape_html_text(url: str, source_name: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = clean_text(" ".join(paragraphs))

    return {
        "source_type": "html",
        "source_name": source_name,
        "url": url,
        "title": clean_text(title),
        "text": text
    }

def fetch_reddit_posts(subreddit: str = "Insurance", limit: int = 8) -> list[dict]:
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    payload = r.json()

    rows = []
    for child in payload.get("data", {}).get("children", []):
        post = child.get("data", {})
        permalink = post.get("permalink", "")
        rows.append({
            "source_type": "forum",
            "source_name": f"reddit:r/{subreddit}",
            "url": "https://www.reddit.com" + permalink if permalink else url,
            "title": clean_text(post.get("title", "")),
            "text": clean_text(post.get("selftext", "")),
        })
    return rows

def main():
    print("\n--- [Step 1] Collect ASCII Text (HTML + Forum) ---")

    html_sources = [
        ("NAIC Consumer", "https://content.naic.org/consumer"),
        ("California Department of Insurance", "https://www.insurance.ca.gov/")
    ]

    all_rows = []

    # HTML pages
    for name, url in html_sources:
        try:
            row = scrape_html_text(url, name)
            all_rows.append(row)
            print(f"[HTML] OK: {name} -> {len(row['text'])} chars")
            time.sleep(1)
        except Exception as e:
            print(f"[HTML] FAIL: {name} ({url}) -> {e}")

    # Forum posts
    try:
        forum_rows = fetch_reddit_posts(subreddit="Insurance", limit=8)
        all_rows.extend(forum_rows)
        print(f"[Forum] OK: fetched {len(forum_rows)} posts")
    except Exception as e:
        print(f"[Forum] FAIL (Reddit may rate-limit): {e}")

    print("\n--- [Step 2] Convert to DataFrame ---")
    df = pd.DataFrame(all_rows)

    print("\n--- [Step 3] Display first 5 records ---")
    print(df.head())

    print("\n--- [Step 4] Dataset dimensions & missing values ---")
    print("Rows, Columns:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())

    print("\n--- [Step 5] Basic text statistics ---")
    df["word_count"] = df["text"].fillna("").apply(lambda x: len(str(x).split()))
    print(df["word_count"].describe())

    out_path = "lab2/data/insurance_web_texts.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved output CSV to: {out_path}")

if __name__ == "__main__":
    main()
