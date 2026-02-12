"""
DSCI 560 Lab 5 - scraper.py
Reddit scraper with two collection methods:
  1. PRAW (official Reddit API)
  2. BeautifulSoup + old.reddit.com (no API key needed)

Usage:
    python scraper.py 5000 --subreddit cybersecurity,netsec,hacking
    python scraper.py 5000 --subreddit cybersecurity,netsec --method bs4
    python scraper.py 5000 --subreddit cybersecurity --init-db
"""

import argparse
import hashlib
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

from dotenv import load_dotenv
load_dotenv()

import requests
from bs4 import BeautifulSoup

# optional PRAW import (not needed for bs4 method)
try:
    import praw
    from praw.models import Submission
    from prawcore.exceptions import (
        TooManyRequests,
        ServerError,
        RequestException,
        ResponseException,
    )
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

from database import get_db, DatabaseManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("scraper")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "YOUR_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT",    "DSCI560Lab5/1.0 (by /u/your_username)")

BATCH_SIZE         = 100
RATE_LIMIT_PAUSE   = 2.0
RETRY_WAIT         = 10
MAX_RETRIES        = 5
DB_FLUSH_EVERY     = 200

# ad / promo keywords for flagging sponsored posts
AD_KEYWORDS = [
    "promoted", "sponsored", "advertisement", "ad", "promo",
    "buy now", "discount", "coupon", "free trial", "sign up",
    "click here", "limited time", "deal", "offer",
]
AD_FLAIR_KEYWORDS = ["promoted", "ad", "sponsor"]

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")


# ===========================================================================
#  Ad detection
# ===========================================================================
def is_advertisement(post: Dict) -> bool:
    """Return True if a post looks like an ad or promotion."""
    # check flair
    flair = (post.get("flair") or "").lower()
    if any(kw in flair for kw in AD_FLAIR_KEYWORDS):
        return True

    # keyword scoring in title + body
    text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()
    ad_score = sum(1 for kw in AD_KEYWORDS if kw in text)
    if ad_score >= 2:
        return True

    # author name heuristic
    author = (post.get("author") or "").lower()
    if author in ("[deleted]", "[removed]"):
        return False
    if any(tag in author for tag in ("official", "_ad", "brand", "promo")):
        return True

    return False


# ===========================================================================
#  Image download
# ===========================================================================
def download_image(url: str, post_id: str) -> Optional[str]:
    """Download image to local images/ directory. Returns path or None."""
    if not url:
        return None

    os.makedirs(IMAGE_DIR, exist_ok=True)

    ext = ".jpg"
    for e in (".png", ".gif", ".webp", ".jpeg"):
        if e in url.lower():
            ext = e
            break

    filename = f"{post_id}{ext}"
    filepath = os.path.join(IMAGE_DIR, filename)

    if os.path.exists(filepath):
        return filepath

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": REDDIT_USER_AGENT})
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            return None
        with open(filepath, "wb") as f:
            f.write(resp.content)
        logger.debug("Image saved: %s", filepath)
        return filepath
    except Exception as e:
        logger.debug("Image download failed (%s): %s", url, e)
        return None


# ===========================================================================
#  Method 1: PRAW (Reddit API)
# ===========================================================================
def create_reddit_client() -> "praw.Reddit":
    if not PRAW_AVAILABLE:
        raise ImportError("praw not installed. Run: pip install praw  or use --method bs4")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    reddit.read_only = True
    logger.info("Reddit PRAW client created (read-only).")
    return reddit


def submission_to_dict(submission: "Submission") -> Dict:
    """Convert a PRAW Submission object to a plain dict."""
    media_url = None
    if hasattr(submission, "preview") and submission.preview:
        try:
            media_url = submission.preview["images"][0]["source"]["url"]
        except (KeyError, IndexError):
            pass
    if media_url is None and submission.url and submission.url.endswith(
        (".jpg", ".jpeg", ".png", ".gif", ".webp")
    ):
        media_url = submission.url

    return {
        "post_id":       submission.id,
        "subreddit":     str(submission.subreddit),
        "title":         submission.title,
        "selftext":      submission.selftext or "",
        "author":        str(submission.author) if submission.author else "[deleted]",
        "score":         submission.score,
        "upvote_ratio":  submission.upvote_ratio,
        "num_comments":  submission.num_comments,
        "created_utc":   submission.created_utc,
        "url":           submission.url,
        "permalink":     submission.permalink,
        "is_self":       submission.is_self,
        "over_18":       submission.over_18,
        "thumbnail":     submission.thumbnail if submission.thumbnail not in ("self", "default", "nsfw", "spoiler", "") else None,
        "media_url":     media_url,
        "flair":         submission.link_flair_text,
    }


def scrape_subreddit_praw(
    reddit: "praw.Reddit",
    subreddit_name: str,
    num_posts: int,
    sort: str = "hot",
    collected: Optional[Dict[str, Dict]] = None,
    db: Optional[DatabaseManager] = None,
    download_images: bool = False,
) -> Dict[str, Dict]:
    """
    Scrape posts from a single subreddit using PRAW.
    Uses multiple sort strategies to maximise unique results.
    """
    if collected is None:
        collected = {}

    subreddit = reddit.subreddit(subreddit_name)
    buffer: List[Dict] = []

    def _pull(generator, tag: str):
        nonlocal buffer
        count_before = len(collected)
        for attempt in range(MAX_RETRIES):
            try:
                for submission in generator:
                    if submission.id in collected:
                        continue
                    post = submission_to_dict(submission)

                    if is_advertisement(post):
                        post["flair"] = (post.get("flair") or "") + " [AD]"
                        logger.debug("Ad detected: %s", post["title"][:60])

                    if download_images and post.get("media_url"):
                        local_path = download_image(post["media_url"], post["post_id"])
                        if local_path:
                            post["thumbnail"] = local_path

                    collected[submission.id] = post
                    buffer.append(post)

                    if db and len(buffer) >= DB_FLUSH_EVERY:
                        db.insert_raw_posts_batch(buffer)
                        buffer = []

                    if len(collected) >= num_posts:
                        break

                    if len(collected) % BATCH_SIZE == 0:
                        time.sleep(RATE_LIMIT_PAUSE)

                break
            except TooManyRequests:
                wait = RETRY_WAIT * (attempt + 1)
                logger.warning("429 Too Many Requests -- waiting %ds", wait)
                time.sleep(wait)
            except (ServerError, RequestException, ResponseException) as e:
                wait = RETRY_WAIT * (attempt + 1)
                logger.warning("Server error (%s) -- retry %d/%d in %ds",
                               e, attempt + 1, MAX_RETRIES, wait)
                time.sleep(wait)

        added = len(collected) - count_before
        logger.info("[r/%s %s] +%d posts  (total %d / %d)",
                    subreddit_name, tag, added, len(collected), num_posts)

    # multi-strategy scraping
    strategies = [
        (sort, lambda: _get_listing(subreddit, sort)),
    ]
    if num_posts > 800:
        extra = [
            ("new",       lambda: subreddit.new(limit=None)),
            ("top_all",   lambda: subreddit.top(time_filter="all", limit=None)),
            ("top_year",  lambda: subreddit.top(time_filter="year", limit=None)),
            ("top_month", lambda: subreddit.top(time_filter="month", limit=None)),
            ("rising",    lambda: subreddit.rising(limit=None)),
        ]
        strategies += [(t, g) for t, g in extra if t != sort]

    for tag, gen_factory in strategies:
        if len(collected) >= num_posts:
            break
        logger.info("r/%s -- Strategy: %s ...", subreddit_name, tag)
        _pull(gen_factory(), tag)

    if db and buffer:
        db.insert_raw_posts_batch(buffer)

    return collected


def _get_listing(subreddit, sort: str):
    sort = sort.lower()
    if sort == "new":
        return subreddit.new(limit=None)
    elif sort == "top":
        return subreddit.top(time_filter="all", limit=None)
    elif sort == "rising":
        return subreddit.rising(limit=None)
    elif sort == "controversial":
        return subreddit.controversial(time_filter="all", limit=None)
    else:
        return subreddit.hot(limit=None)


# ===========================================================================
#  Method 2: BeautifulSoup + old.reddit.com (no API key needed)
# ===========================================================================
def scrape_subreddit_bs4(
    subreddit_name: str,
    num_posts: int,
    collected: Optional[Dict[str, Dict]] = None,
    db: Optional[DatabaseManager] = None,
    download_images: bool = False,
) -> Dict[str, Dict]:
    """
    Scrape posts from old.reddit.com using BeautifulSoup.
    No API key required; paginates via the 'after' token.
    """
    if collected is None:
        collected = {}

    base_url = f"https://old.reddit.com/r/{subreddit_name}"
    headers = {"User-Agent": REDDIT_USER_AGENT}
    buffer: List[Dict] = []
    after = None
    page = 0

    while len(collected) < num_posts:
        page += 1
        url = base_url
        params = {}
        if after:
            params["after"] = after

        # request with retry
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                break
            except Exception as e:
                wait = RETRY_WAIT * (attempt + 1)
                logger.warning("BS4 request error (%s) -- retry in %ds", e, wait)
                time.sleep(wait)
        else:
            logger.error("BS4: max retries exceeded on page %d, stopping.", page)
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        things = soup.find_all("div", class_="thing", attrs={"data-fullname": True})
        if not things:
            logger.info("BS4: no more posts on page %d", page)
            break

        for thing in things:
            # skip promoted posts
            if "promoted" in thing.get("class", []):
                logger.debug("BS4: skipping promoted post")
                continue

            post_id = thing.get("data-fullname", "").replace("t3_", "")
            if not post_id or post_id in collected:
                continue

            # extract fields
            title_tag = thing.find("a", class_="title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            post_url = title_tag.get("href", "") if title_tag else ""
            if post_url.startswith("/"):
                post_url = urljoin("https://old.reddit.com", post_url)

            score_tag = thing.find("div", class_="score unvoted")
            score_text = score_tag.get("title", "0") if score_tag else "0"
            try:
                score = int(score_text)
            except ValueError:
                score = 0

            author_tag = thing.find("a", class_="author")
            author = author_tag.get_text(strip=True) if author_tag else "[deleted]"

            comments_tag = thing.find("a", class_="comments")
            num_comments = 0
            if comments_tag:
                c_text = comments_tag.get_text(strip=True)
                c_match = re.search(r"(\d+)", c_text)
                if c_match:
                    num_comments = int(c_match.group(1))

            time_tag = thing.find("time")
            created_utc = 0
            if time_tag and time_tag.get("datetime"):
                try:
                    dt = datetime.fromisoformat(time_tag["datetime"].replace("Z", "+00:00"))
                    created_utc = dt.timestamp()
                except Exception:
                    pass

            permalink = thing.get("data-permalink", "")

            flair_tag = thing.find("span", class_="linkflairlabel")
            flair = flair_tag.get_text(strip=True) if flair_tag else None

            # check for images
            thumb_tag = thing.find("a", class_="thumbnail")
            media_url = None
            thumbnail = None
            if thumb_tag:
                img = thumb_tag.find("img")
                if img and img.get("src"):
                    thumbnail = img["src"]
            if post_url and any(post_url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
                media_url = post_url
            if "i.redd.it" in post_url or "i.imgur.com" in post_url:
                media_url = post_url

            is_self = thing.get("data-domain", "") == f"self.{subreddit_name}"

            post = {
                "post_id":       post_id,
                "subreddit":     subreddit_name,
                "title":         title,
                "selftext":      "",  # listing page doesn't include body text
                "author":        author,
                "score":         score,
                "upvote_ratio":  0,
                "num_comments":  num_comments,
                "created_utc":   created_utc,
                "url":           post_url,
                "permalink":     permalink,
                "is_self":       is_self,
                "over_18":       "nsfw" in thing.get("class", []),
                "thumbnail":     thumbnail,
                "media_url":     media_url,
                "flair":         flair,
            }

            if is_advertisement(post):
                post["flair"] = (post.get("flair") or "") + " [AD]"

            if download_images and media_url:
                local_path = download_image(media_url, post_id)
                if local_path:
                    post["thumbnail"] = local_path

            collected[post_id] = post
            buffer.append(post)

            if db and len(buffer) >= DB_FLUSH_EVERY:
                db.insert_raw_posts_batch(buffer)
                buffer = []

            if len(collected) >= num_posts:
                break

        # pagination
        next_btn = soup.find("span", class_="next-button")
        if next_btn:
            next_link = next_btn.find("a")
            if next_link and next_link.get("href"):
                next_url = next_link["href"]
                after_match = re.search(r"after=([^&]+)", next_url)
                after = after_match.group(1) if after_match else None
            else:
                after = None
        else:
            after = None

        if after is None:
            logger.info("BS4: no more pages for r/%s (page %d)", subreddit_name, page)
            break

        logger.info("BS4 r/%s -- page %d done, total %d / %d",
                    subreddit_name, page, len(collected), num_posts)
        time.sleep(RATE_LIMIT_PAUSE)

    if db and buffer:
        db.insert_raw_posts_batch(buffer)

    return collected


# ===========================================================================
#  Multi-subreddit orchestrator
# ===========================================================================
def scrape_multiple_subreddits(
    subreddit_names: List[str],
    num_posts: int,
    sort: str = "hot",
    method: str = "praw",
    db: Optional[DatabaseManager] = None,
    download_images: bool = False,
) -> List[Dict]:
    """
    Scrape posts from multiple subreddits with global deduplication.
    Quota is split evenly; shortfalls carry over to the next subreddit.
    """
    collected: Dict[str, Dict] = {}
    reddit = None

    if method == "praw":
        reddit = create_reddit_client()

    for i, sub in enumerate(subreddit_names):
        remaining = num_posts - len(collected)
        if remaining <= 0:
            break

        subs_left = len(subreddit_names) - i
        per_sub = max(100, remaining // subs_left)
        target = min(per_sub, remaining)

        logger.info("=" * 50)
        logger.info("Scraping r/%s  (target: %d, total so far: %d / %d)",
                    sub, target, len(collected), num_posts)
        logger.info("=" * 50)

        log_id = None
        if db:
            log_id = db.log_scrape_start(sub, target)

        before_count = len(collected)

        if method == "praw":
            collected = scrape_subreddit_praw(
                reddit, sub, len(collected) + target,
                sort=sort, collected=collected, db=db,
                download_images=download_images,
            )
        else:
            collected = scrape_subreddit_bs4(
                sub, len(collected) + target,
                collected=collected, db=db,
                download_images=download_images,
            )

        scraped = len(collected) - before_count
        if db and log_id:
            db.log_scrape_end(log_id, scraped, "success")

        logger.info("r/%s done: +%d posts (total %d)", sub, scraped, len(collected))

    return list(collected.values())


# ===========================================================================
#  CLI
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Reddit Scraper - DSCI 560 Lab 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py 5000 --subreddit cybersecurity,netsec,hacking
  python scraper.py 5000 --subreddit cybersecurity --method bs4
  python scraper.py 5000 --subreddit cybersecurity --init-db --download-images
        """,
    )
    parser.add_argument(
        "num_posts", type=int,
        help="Total number of posts to scrape (e.g. 5000)",
    )
    parser.add_argument(
        "--subreddit", "-s", type=str, default="cybersecurity",
        help="Subreddit name(s), comma-separated (e.g. cybersecurity,netsec,hacking)",
    )
    parser.add_argument(
        "--sort", type=str, default="hot",
        choices=["hot", "new", "top", "rising", "controversial"],
        help="Initial sort order (default: hot)",
    )
    parser.add_argument(
        "--method", "-m", type=str, default="praw",
        choices=["praw", "bs4"],
        help="Scraping method: praw (API) or bs4 (old.reddit.com)",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip database writes, print to console only (debug mode)",
    )
    parser.add_argument(
        "--init-db", action="store_true",
        help="Initialise the database before scraping (runs setup_db.sql)",
    )
    parser.add_argument(
        "--download-images", action="store_true",
        help="Download post images to local images/ directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    subreddits = [s.strip() for s in args.subreddit.split(",") if s.strip()]

    db: Optional[DatabaseManager] = None
    if not args.no_db:
        db = get_db()
        if args.init_db:
            sql_path = os.path.join(os.path.dirname(__file__), "setup_db.sql")
            db.init_database(sql_path)
            logger.info("Database initialised.")

    start_time = time.time()

    print(f"\nStarting Reddit scrape")
    print(f"  Target:     {args.num_posts}")
    print(f"  Subreddits: {', '.join('r/' + s for s in subreddits)}")
    print(f"  Method:     {args.method.upper()}")
    print(f"  Images:     {'yes' if args.download_images else 'no'}")
    print()

    posts = scrape_multiple_subreddits(
        subreddit_names=subreddits,
        num_posts=args.num_posts,
        sort=args.sort,
        method=args.method,
        db=db,
        download_images=args.download_images,
    )

    elapsed = time.time() - start_time

    # summary stats
    ad_count = sum(1 for p in posts if "[AD]" in (p.get("flair") or ""))
    img_count = sum(1 for p in posts if p.get("media_url"))
    sub_counts = {}
    for p in posts:
        sub_counts[p["subreddit"]] = sub_counts.get(p["subreddit"], 0) + 1

    print(f"\n{'='*60}")
    print(f"  Scrape complete!")
    print(f"  Target:       {args.num_posts}")
    print(f"  Collected:    {len(posts)}")
    print(f"  Ads detected: {ad_count}")
    print(f"  With images:  {img_count}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  -- Per-subreddit breakdown --")
    for sub, cnt in sorted(sub_counts.items(), key=lambda x: -x[1]):
        print(f"    r/{sub}: {cnt}")
    if db:
        total = db.get_total_post_count()
        print(f"  DB total:     {total}")
    print(f"{'='*60}\n")

    # preview first 5
    for i, p in enumerate(posts[:5]):
        ad_tag = " [AD]" if "[AD]" in (p.get("flair") or "") else ""
        img_tag = " [IMG]" if p.get("media_url") else ""
        print(f"[{i+1}] r/{p['subreddit']} | {p['title'][:70]}{ad_tag}{img_tag}")
        print(f"    score={p['score']}  comments={p['num_comments']}  author={p['author']}")
        print()


if __name__ == "__main__":
    main()
