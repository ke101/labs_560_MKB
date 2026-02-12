"""
DSCI 560 Lab 5 - database.py
MySQL database operations module.
Provides connection management, table init, and CRUD operations.
"""

import os
import json
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
load_dotenv()

import mysql.connector
from mysql.connector import Error, pooling
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config (overridable via environment variables)
# ---------------------------------------------------------------------------
DEFAULT_DB_CONFIG = {
    "host":     os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port":     int(os.getenv("MYSQL_PORT", 3306)),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "reddit_db"),
    "charset":  "utf8mb4",
    "collation": "utf8mb4_unicode_ci",
}


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------
class DatabaseManager:
    """Encapsulates all MySQL operations for the Reddit pipeline."""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None, pool_size: int = 5):
        self.config = db_config or DEFAULT_DB_CONFIG
        self.pool_size = pool_size
        self._pool: Optional[pooling.MySQLConnectionPool] = None

    # ------------------------------------------------------------------ #
    # Connection pool
    # ------------------------------------------------------------------ #
    def _get_pool(self) -> pooling.MySQLConnectionPool:
        if self._pool is None:
            self._pool = pooling.MySQLConnectionPool(
                pool_name="reddit_pool",
                pool_size=self.pool_size,
                pool_reset_session=True,
                **self.config,
            )
            logger.info("MySQL connection pool created (size=%d)", self.pool_size)
        return self._pool

    @contextmanager
    def get_connection(self):
        """Acquire a pooled database connection (context manager)."""
        conn = self._get_pool().get_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def get_cursor(self, dictionary: bool = True, commit: bool = True):
        """Get a cursor with auto-commit and auto-close."""
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=dictionary)
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    # ------------------------------------------------------------------ #
    # Database & table initialisation
    # ------------------------------------------------------------------ #
    def init_database(self, sql_file: str = "setup_db.sql"):
        """Execute setup_db.sql to create the database and all tables."""
        cfg = {k: v for k, v in self.config.items() if k != "database"}
        conn = mysql.connector.connect(**cfg)
        cursor = conn.cursor()
        try:
            with open(sql_file, "r", encoding="utf-8") as f:
                sql_script = f.read()
            for statement in sql_script.split(";"):
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt)
            conn.commit()
            logger.info("Database initialised from %s", sql_file)
        finally:
            cursor.close()
            conn.close()

    # ------------------------------------------------------------------ #
    # raw_posts operations
    # ------------------------------------------------------------------ #
    def insert_raw_post(self, post: Dict[str, Any]) -> bool:
        """Insert a single raw post. Returns True if new, False if updated."""
        sql = """
            INSERT INTO raw_posts
                (post_id, subreddit, title, selftext, author, score,
                 upvote_ratio, num_comments, created_utc, url, permalink,
                 is_self, over_18, thumbnail, media_url, flair)
            VALUES
                (%(post_id)s, %(subreddit)s, %(title)s, %(selftext)s,
                 %(author)s, %(score)s, %(upvote_ratio)s, %(num_comments)s,
                 %(created_utc)s, %(url)s, %(permalink)s, %(is_self)s,
                 %(over_18)s, %(thumbnail)s, %(media_url)s, %(flair)s)
            ON DUPLICATE KEY UPDATE
                score        = VALUES(score),
                upvote_ratio = VALUES(upvote_ratio),
                num_comments = VALUES(num_comments),
                scraped_at   = CURRENT_TIMESTAMP
        """
        with self.get_cursor() as cur:
            cur.execute(sql, post)
            return cur.rowcount == 1

    def insert_raw_posts_batch(self, posts: List[Dict[str, Any]]) -> int:
        """Batch-insert raw posts. Returns affected row count."""
        if not posts:
            return 0
        sql = """
            INSERT INTO raw_posts
                (post_id, subreddit, title, selftext, author, score,
                 upvote_ratio, num_comments, created_utc, url, permalink,
                 is_self, over_18, thumbnail, media_url, flair)
            VALUES
                (%(post_id)s, %(subreddit)s, %(title)s, %(selftext)s,
                 %(author)s, %(score)s, %(upvote_ratio)s, %(num_comments)s,
                 %(created_utc)s, %(url)s, %(permalink)s, %(is_self)s,
                 %(over_18)s, %(thumbnail)s, %(media_url)s, %(flair)s)
            ON DUPLICATE KEY UPDATE
                score        = VALUES(score),
                upvote_ratio = VALUES(upvote_ratio),
                num_comments = VALUES(num_comments),
                scraped_at   = CURRENT_TIMESTAMP
        """
        with self.get_cursor() as cur:
            cur.executemany(sql, posts)
            affected = cur.rowcount
        logger.info("Batch insert: %d rows affected (%d posts)", affected, len(posts))
        return affected

    def get_raw_posts(
        self,
        subreddit: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """Query raw posts with optional subreddit filter."""
        sql = "SELECT * FROM raw_posts"
        params: list = []
        if subreddit:
            sql += " WHERE subreddit = %s"
            params.append(subreddit)
        sql += " ORDER BY created_utc DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        with self.get_cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def get_unprocessed_posts(self, limit: int = 500) -> List[Dict]:
        """Get posts that haven't been cleaned yet."""
        sql = """
            SELECT r.* FROM raw_posts r
            LEFT JOIN cleaned_posts c ON r.post_id = c.post_id
            WHERE c.post_id IS NULL
            ORDER BY r.created_utc DESC
            LIMIT %s
        """
        with self.get_cursor() as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()

    def get_total_post_count(self, subreddit: Optional[str] = None) -> int:
        """Return total post count."""
        sql = "SELECT COUNT(*) AS cnt FROM raw_posts"
        params: list = []
        if subreddit:
            sql += " WHERE subreddit = %s"
            params.append(subreddit)
        with self.get_cursor(commit=False) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row["cnt"] if row else 0

    # ------------------------------------------------------------------ #
    # cleaned_posts operations
    # ------------------------------------------------------------------ #
    def insert_cleaned_post(self, post: Dict[str, Any]) -> bool:
        sql = """
            INSERT INTO cleaned_posts
                (post_id, subreddit, title_clean, body_clean, author_anon,
                 score, num_comments, created_dt, url, is_ad,
                 ocr_text, keywords, topic)
            VALUES
                (%(post_id)s, %(subreddit)s, %(title_clean)s, %(body_clean)s,
                 %(author_anon)s, %(score)s, %(num_comments)s, %(created_dt)s,
                 %(url)s, %(is_ad)s, %(ocr_text)s, %(keywords)s, %(topic)s)
            ON DUPLICATE KEY UPDATE
                title_clean  = VALUES(title_clean),
                body_clean   = VALUES(body_clean),
                keywords     = VALUES(keywords),
                topic        = VALUES(topic),
                processed_at = CURRENT_TIMESTAMP
        """
        with self.get_cursor() as cur:
            cur.execute(sql, post)
            return cur.rowcount >= 1

    def insert_cleaned_posts_batch(self, posts: List[Dict[str, Any]]) -> int:
        if not posts:
            return 0
        sql = """
            INSERT INTO cleaned_posts
                (post_id, subreddit, title_clean, body_clean, author_anon,
                 score, num_comments, created_dt, url, is_ad,
                 ocr_text, keywords, topic)
            VALUES
                (%(post_id)s, %(subreddit)s, %(title_clean)s, %(body_clean)s,
                 %(author_anon)s, %(score)s, %(num_comments)s, %(created_dt)s,
                 %(url)s, %(is_ad)s, %(ocr_text)s, %(keywords)s, %(topic)s)
            ON DUPLICATE KEY UPDATE
                title_clean  = VALUES(title_clean),
                body_clean   = VALUES(body_clean),
                keywords     = VALUES(keywords),
                topic        = VALUES(topic),
                processed_at = CURRENT_TIMESTAMP
        """
        with self.get_cursor() as cur:
            cur.executemany(sql, posts)
            return cur.rowcount

    def get_cleaned_posts(self, limit: int = 100) -> List[Dict]:
        sql = "SELECT * FROM cleaned_posts ORDER BY created_dt DESC LIMIT %s"
        with self.get_cursor(commit=False) as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()

    # ------------------------------------------------------------------ #
    # post_vectors operations
    # ------------------------------------------------------------------ #
    def insert_vector(self, post_id: str, vector: np.ndarray, model_name: str = "doc2vec"):
        """Store an embedding vector for a post."""
        sql = """
            INSERT INTO post_vectors (post_id, vector, vector_dim, model_name)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                vector     = VALUES(vector),
                vector_dim = VALUES(vector_dim),
                model_name = VALUES(model_name),
                created_at = CURRENT_TIMESTAMP
        """
        blob = vector.astype(np.float32).tobytes()
        with self.get_cursor() as cur:
            cur.execute(sql, (post_id, blob, vector.shape[0], model_name))

    def insert_vectors_batch(self, records: List[Dict[str, Any]]):
        """Batch-store vectors. Each record needs post_id, vector, model_name."""
        sql = """
            INSERT INTO post_vectors (post_id, vector, vector_dim, model_name)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                vector     = VALUES(vector),
                vector_dim = VALUES(vector_dim),
                model_name = VALUES(model_name),
                created_at = CURRENT_TIMESTAMP
        """
        rows = []
        for r in records:
            v = r["vector"].astype(np.float32)
            rows.append((r["post_id"], v.tobytes(), v.shape[0], r.get("model_name", "doc2vec")))
        with self.get_cursor() as cur:
            cur.executemany(sql, rows)

    def get_all_vectors(self) -> List[Dict]:
        """Read all vectors (deserialised to numpy arrays)."""
        sql = "SELECT post_id, vector, vector_dim, model_name FROM post_vectors"
        with self.get_cursor(commit=False) as cur:
            cur.execute(sql)
            results = []
            for row in cur.fetchall():
                vec = np.frombuffer(row["vector"], dtype=np.float32)
                results.append({
                    "post_id":    row["post_id"],
                    "vector":     vec,
                    "vector_dim": row["vector_dim"],
                    "model_name": row["model_name"],
                })
            return results

    # ------------------------------------------------------------------ #
    # cluster_results / cluster_keywords operations
    # ------------------------------------------------------------------ #
    def save_cluster_results(self, run_id: int, assignments: List[Dict[str, Any]]):
        """Save clustering assignments for one run."""
        sql = """
            INSERT INTO cluster_results
                (post_id, cluster_id, distance_to_center, is_nearest, run_id)
            VALUES
                (%(post_id)s, %(cluster_id)s, %(distance_to_center)s,
                 %(is_nearest)s, %(run_id)s)
        """
        with self.get_cursor() as cur:
            cur.executemany(sql, [{**a, "run_id": run_id} for a in assignments])

    def save_cluster_keywords(self, run_id: int, keywords: List[Dict[str, Any]]):
        """Save keywords for each cluster."""
        sql = """
            INSERT INTO cluster_keywords
                (cluster_id, run_id, keyword, weight)
            VALUES
                (%(cluster_id)s, %(run_id)s, %(keyword)s, %(weight)s)
        """
        with self.get_cursor() as cur:
            cur.executemany(sql, [{**k, "run_id": run_id} for k in keywords])

    def get_latest_run_id(self) -> int:
        """Get the latest clustering run_id, or 0 if none."""
        sql = "SELECT COALESCE(MAX(run_id), 0) AS max_run FROM cluster_results"
        with self.get_cursor(commit=False) as cur:
            cur.execute(sql)
            row = cur.fetchone()
            return row["max_run"] if row else 0

    def get_cluster_posts(self, cluster_id: int, run_id: Optional[int] = None, limit: int = 20) -> List[Dict]:
        """Query posts in a given cluster."""
        if run_id is None:
            run_id = self.get_latest_run_id()
        sql = """
            SELECT cr.cluster_id, cr.distance_to_center, cr.is_nearest,
                   cp.post_id, cp.title_clean, cp.body_clean, cp.keywords
            FROM cluster_results cr
            JOIN cleaned_posts cp ON cr.post_id = cp.post_id
            WHERE cr.cluster_id = %s AND cr.run_id = %s
            ORDER BY cr.distance_to_center ASC
            LIMIT %s
        """
        with self.get_cursor(commit=False) as cur:
            cur.execute(sql, (cluster_id, run_id, limit))
            return cur.fetchall()

    def search_clusters_by_keyword(self, keyword: str, run_id: Optional[int] = None) -> List[Dict]:
        """Search clusters matching a keyword."""
        if run_id is None:
            run_id = self.get_latest_run_id()
        sql = """
            SELECT ck.cluster_id, ck.keyword, ck.weight
            FROM cluster_keywords ck
            WHERE ck.run_id = %s AND ck.keyword LIKE %s
            ORDER BY ck.weight DESC
        """
        with self.get_cursor(commit=False) as cur:
            cur.execute(sql, (run_id, f"%{keyword}%"))
            return cur.fetchall()

    # ------------------------------------------------------------------ #
    # scrape_log operations
    # ------------------------------------------------------------------ #
    def log_scrape_start(self, subreddit: str, posts_requested: int) -> int:
        """Log the start of a scrape run. Returns log_id."""
        sql = """
            INSERT INTO scrape_log (subreddit, posts_requested, started_at, status)
            VALUES (%s, %s, %s, 'running')
        """
        with self.get_cursor() as cur:
            cur.execute(sql, (subreddit, posts_requested, datetime.utcnow()))
            return cur.lastrowid

    def log_scrape_end(self, log_id: int, posts_scraped: int, status: str = "success", error: str = None):
        sql = """
            UPDATE scrape_log
            SET posts_scraped = %s, finished_at = %s, status = %s, error_message = %s
            WHERE id = %s
        """
        with self.get_cursor() as cur:
            cur.execute(sql, (posts_scraped, datetime.utcnow(), status, error, log_id))

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def execute_query(self, sql: str, params: tuple = (), fetch: bool = True) -> Any:
        """Run an arbitrary query."""
        with self.get_cursor(commit=not fetch) as cur:
            cur.execute(sql, params)
            if fetch:
                return cur.fetchall()

    def close(self):
        """Close the connection pool."""
        self._pool = None
        logger.info("Database manager closed.")


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_instance: Optional[DatabaseManager] = None


def get_db(config: Optional[Dict] = None) -> DatabaseManager:
    """Get the global DatabaseManager singleton."""
    global _instance
    if _instance is None:
        _instance = DatabaseManager(config)
    return _instance


# ---------------------------------------------------------------------------
# CLI: run directly to initialise the database
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    sql_path = os.path.join(os.path.dirname(__file__), "setup_db.sql")
    if not os.path.exists(sql_path):
        print(f"Error: cannot find {sql_path}")
        sys.exit(1)

    db = get_db()
    db.init_database(sql_path)
    print("Database and tables initialised successfully.")
