import sys
import time
import threading
import subprocess
import argparse
from datetime import datetime
import schedule
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter
import signal

### !!!!! some functions in this file are not used in final pipline yet, cuz till bugs in it, but will do optimize after
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'your_username',
    'password': 'your_password',
    'database': 'reddit_data'
}

class PipelineOrchestrator:
    def __init__(self, interval_minutes, data_number='100', method='bs4'):
        """
        Initialize the pipeline orchestrator
        
        Args:
            interval_minutes: How often to run the pipeline (in minutes)
            data_number: Number of posts to scrape
            method: Scraping method (bs4 or api)
        """
        self.interval = interval_minutes
        self.data_number = data_number
        self.method = method
        self.is_running = True
        self.last_run = None
        self.pipeline_thread = None
        
    def run_scraping(self):
        print(f"STAGE 1: Scraping Data")
        
        try:
            cmd = ['python', 'scrape.py', self.data_number, '--method', self.method]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("Scraping completed successfully")
                print(f"Collected {self.data_number} posts")
                return True
            else:
                print("Scraping failed")
                print(f"  Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Scraping timed out")
            return False
        except FileNotFoundError:
            print("scrape.py not found in current directory")
            return False
        except Exception as e:
            print(f"Unexpected error during scraping: {str(e)}")
            return False
    
    def run_preprocessing(self):
        """Execute the preprocessing and vectorization stage"""
        print(f"STAGE 2: Preprocessing & Vectorization")
        
        try:
            print("Running preprocessing.py")
            result = subprocess.run(['python', 'preprocessing.py'], 
                                  capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                print("Preprocessing failed")
                print(f"  Error: {result.stderr}")
                return False
            
            print("Preprocessing completed")
            
            # Run vectorization
            print("Running vectorize.py")
            result = subprocess.run(['python', 'vectorize.py'], 
                                  capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                print("Vectorization failed")
                print(f"  Error: {result.stderr}")
                return False
            
            print("Vectorization completed")
            print(" Text cleaned and embedded")
            return True
            
        except subprocess.TimeoutExpired:
            print("Preprocessing/Vectorization timed out")
            return False
        except FileNotFoundError as e:
            print(f"Required script not found: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error during preprocessing: {str(e)}")
            return False
    
    def run_clustering(self):
        """Execute the clustering stage"""
        print(f"STAGE 3: Clustering")
        
        try:
            # Get MySQL credentials
            username = MYSQL_CONFIG['user']
            password = MYSQL_CONFIG['password']
            
            print("Running cluster script")
            cmd = ['python', 'cluster', username, password]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("Clustering completed")
                print("Clusters identified and stored in database")
                return True
            else:
                print("Clustering failed")
                print(f" Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Clustering timed out (exceeded 5 minutes)")
            return False
        except FileNotFoundError:
            print("cluster script not found")
            return False
        except Exception as e:
            print(f"Unexpected error during clustering: {str(e)}")
            return False
    
    def run_keyword_extraction(self):
        """Execute the keyword extraction stage"""
        print(f"STAGE 4: Keyword Extraction")
        
        try:
            username = MYSQL_CONFIG['user']
            password = MYSQL_CONFIG['password']
            
            print("Running keywords script")
            cmd = ['python', 'keywords', username, password]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("Keyword extraction completed")
                print("Keywords generated for each cluster")
                return True
            else:
                print("Keyword extraction failed")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Keyword extraction timed out (exceeded 5 minutes)")
            return False
        except FileNotFoundError:
            print("keywords script not found")
            return False
        except Exception as e:
            print(f"Unexpected error during keyword extraction: {str(e)}")
            return False
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        start_time = datetime.now()
        print(f"PIPELINE EXECUTION STARTED")

        
        if not self.run_scraping():
            print("PIPELINE ABORTED: Scraping failed")
            return False
        
        if not self.run_preprocessing():
            print("! PIPELINE ABORTED: Preprocessing failed")
            return False
    
        if not self.run_clustering():
            print("PIPELINE ABORTED: Clustering failed")
            return False
    
        if not self.run_keyword_extraction():
            print("PIPELINE ABORTED: Keyword extraction failed")
            return False
        
        # Success
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Next update in: {self.interval} minutes")

        
        self.last_run = end_time
        return True
    
    def schedule_pipeline(self):
        """Schedule the pipeline to run at specified intervals"""
        # Run immediately on start
        self.run_full_pipeline()
        
        # Schedule future runs
        schedule.every(self.interval).minutes.do(self.run_full_pipeline)
        
        # Keep running scheduled tasks
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def start_background_updates(self):
        """Start the pipeline in a background thread"""
        self.pipeline_thread = threading.Thread(target=self.schedule_pipeline, daemon=True)
        self.pipeline_thread.start()
        print(f"\nBackground updates started (every {self.interval} minutes)")
        print("  Type keywords to search clusters, or 'quit' to exit\n")
    
    def stop(self):
        """Stop the pipeline"""
        self.is_running = False
        schedule.clear()


class ClusterSearch:
    def __init__(self):
        """Initialize cluster search functionality"""
        self.clusters = None
        self.cluster_keywords = None
        self.cluster_vectors = None
        
    def load_clusters_from_db(self):
        """Load clusters and keywords from database"""
        try:
            import mysql.connector
            
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            # Load clusters with their keywords
            query = """
                SELECT cluster_id, keywords
                FROM  cluster_keyword
                ORDER BY cluster_id
            """
            cursor.execute(query)
            results = cursor.fetchall()
            
            self.clusters = {}
            self.cluster_keywords = {}
            
            for row in results:
                cluster_id = row['cluster_id']
                keywords = row['keywords'].split(',') if row['keywords'] else []
                self.clusters[cluster_id] = {
                    'keywords': keywords,
                    'count': row['posts_count']
                }
                self.cluster_keywords[cluster_id] = ' '.join(keywords)
            
            cursor.close()
            conn.close()
            
            return len(self.clusters) > 0
            
        except Exception as e:
            print(f"Error loading clusters: {str(e)}")
            return False
    
    def find_matching_cluster(self, query):
        """Find the cluster that best matches the query"""
        if not self.clusters:
            if not self.load_clusters_from_db():
                return None
        
        query_vector, cluster_vectors = self.vectorize_query(query)
        similarities = cosine_similarity(query_vector, cluster_vectors)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        cluster_ids = list(self.clusters.keys())
        
        if best_score < 0.1:  # threshold added but not used yet
            return None
        
        return cluster_ids[best_idx], best_score
    
    def get_cluster_posts(self, cluster_id, limit=10):
        """Retrieve posts from a specific cluster"""
        try:
            import mysql.connector
            
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT title, author, score, created_date, url
                FROM raw_posts
                WHERE cluster_id = %s
                ORDER BY score DESC
                LIMIT %s
            """
            cursor.execute(query, (cluster_id, limit))
            posts = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return posts
            
        except Exception as e:
            print(f"Error retrieving posts: {str(e)}")
            return []
    
    def display_cluster_results(self, cluster_id, score):
        """Display cluster information and posts"""
        cluster = self.clusters[cluster_id]
        

        print(f"CLUSTER #{cluster_id} (Match Score: {score:.2%})")
        print(f"Keywords: {', '.join(cluster['keywords'][:10])}")
        print(f"Total Posts: {cluster['count']}")
        print(f"\nTop Posts:")
        
        posts = self.get_cluster_posts(cluster_id, limit=10)
        
        for i, post in enumerate(posts, 1):
            print(f"\n{i}. {post['title']}")
            print(f"   Author: {post['author']} | Score: {post['score']} | Date: {post['created_date']}")
            print(f"   URL: {post['url']}")
        
        # Create visualization
        # self.visualize_cluster(cluster_id, posts)
    



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Reddit Data Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('interval', type=int,
                       help='Update interval in minutes')
    parser.add_argument('--posts', type=str, default='100',
                       help='Number of posts to scrape (default: 100)')
    parser.add_argument('--method', type=str, choices=['bs4', 'api'], 
                       default='bs4',
                       help='Scraping method (default: bs4)')
    
    args = parser.parse_args()
    
    # Validate interval
    if args.interval < 1:
        print("Error: Interval must be at least 1 minute")
        sys.exit(1)
    
    # Print configuration
    print("REDDIT DATA PIPELINE")
    print(f"Update Interval: {args.interval} minutes")
    print(f"Posts per scrape: {args.posts}")
    print(f"Scraping method: {args.method}")
    # Initialize components
    orchestrator = PipelineOrchestrator(args.interval, args.posts, args.method)
    searcher = ClusterSearch()
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal. Shutting down...")
        orchestrator.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    orchestrator.start_background_updates()
    interactive_mode(orchestrator, searcher)
    
    print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
    