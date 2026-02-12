-- ============================================================
-- DSCI 560 Lab 5 - Reddit Data Pipeline
-- Database Schema Setup Script
-- ============================================================

CREATE DATABASE IF NOT EXISTS reddit_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE reddit_db;

-- ------------------------------------------------------------
-- 1. Raw posts table (scraped from Reddit)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_posts (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    post_id         VARCHAR(20)  NOT NULL UNIQUE,       -- Reddit post id (e.g. "t3_xxxxx")
    subreddit       VARCHAR(100) NOT NULL,
    title           TEXT,
    selftext        LONGTEXT,                            -- body text (may contain HTML)
    author          VARCHAR(100),
    score           INT          DEFAULT 0,
    upvote_ratio    FLOAT        DEFAULT 0,
    num_comments    INT          DEFAULT 0,
    created_utc     DOUBLE,                              -- Unix timestamp
    url             TEXT,
    permalink       TEXT,
    is_self         BOOLEAN      DEFAULT TRUE,
    over_18         BOOLEAN      DEFAULT FALSE,
    thumbnail       TEXT,
    media_url       TEXT,                                 -- direct image / video link
    flair           VARCHAR(255),
    scraped_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_subreddit (subreddit),
    INDEX idx_created   (created_utc),
    INDEX idx_score     (score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 2. Cleaned posts table (preprocessed data)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cleaned_posts (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    post_id         VARCHAR(20)  NOT NULL UNIQUE,
    subreddit       VARCHAR(100) NOT NULL,
    title_clean     TEXT,
    body_clean      LONGTEXT,                            -- cleaned body text
    author_anon     VARCHAR(64),                         -- anonymised username hash
    score           INT          DEFAULT 0,
    num_comments    INT          DEFAULT 0,
    created_dt      DATETIME,                            -- converted from unix ts
    url             TEXT,
    is_ad           BOOLEAN      DEFAULT FALSE,          -- flagged as ad/promo
    ocr_text        LONGTEXT,                            -- text extracted from images
    keywords        TEXT,                                 -- comma-separated keywords
    topic           VARCHAR(255),                        -- identified topic label
    processed_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES raw_posts(post_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_cp_subreddit (subreddit),
    INDEX idx_cp_created   (created_dt)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 3. Post vectors table (embeddings / doc2vec)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS post_vectors (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    post_id         VARCHAR(20)  NOT NULL UNIQUE,
    vector          LONGBLOB     NOT NULL,               -- serialised numpy array
    vector_dim      INT          NOT NULL,               -- dimensionality (e.g. 100)
    model_name      VARCHAR(100) DEFAULT 'doc2vec',      -- which model produced it
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES raw_posts(post_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 4. Clustering results table
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cluster_results (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    post_id         VARCHAR(20)  NOT NULL,
    cluster_id      INT          NOT NULL,
    distance_to_center FLOAT,                            -- distance to cluster centroid
    is_nearest      BOOLEAN      DEFAULT FALSE,          -- nearest msg to centroid?
    run_id          INT          NOT NULL,                -- which clustering run
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES raw_posts(post_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_cr_cluster (cluster_id),
    INDEX idx_cr_run     (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 5. Cluster keywords table
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cluster_keywords (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    cluster_id      INT          NOT NULL,
    run_id          INT          NOT NULL,
    keyword         VARCHAR(255) NOT NULL,
    weight          FLOAT        DEFAULT 0,              -- TF-IDF weight or similar
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_ck_cluster (cluster_id),
    INDEX idx_ck_run     (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 6. Scrape log table (for automation tracking)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scrape_log (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    subreddit       VARCHAR(100) NOT NULL,
    posts_requested INT,
    posts_scraped   INT,
    started_at      DATETIME,
    finished_at     DATETIME,
    status          ENUM('running','success','failed') DEFAULT 'running',
    error_message   TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
