"""
Configuration for the learning path recommender system
"""

import os

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Simulation parameters
NUM_USERS = 50
NUM_ITEMS = 30
STEPS_PER_USER = 20
RANDOM_SEED = 42

# Feature configuration
FEATURE_COLUMNS = [
    'skill_gap',
    'fraction_skills_mastered',
    'difficulty_gap',
    'user_success_rate',
    'user_avg_quiz',
    'user_avg_time',
    'user_num_attempts',
    'item_avg_success',
    'item_avg_quiz',
    'item_avg_time',
    'item_num_skills',
    'skill_match',
    'difficulty',
    'estimated_time'
]

# Ranking model parameters
RANKING_MODEL_TYPE = 'random_forest'  # 'random_forest' or 'ridge'
RANKING_MODEL_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_SEED,
    'max_depth': 10,
}

# Recommendation parameters
TOP_K = 5  # Number of recommendations to return per user
MIN_RELEVANCE_THRESHOLD = 0.2  # Minimum relevance score to recommend

# Relevance formula weights
RELEVANCE_WEIGHTS = {
    'skill_gain': 0.5,           # 50% - how much user learns
    'challenge_alignment': 0.3,  # 30% - how well-suited the difficulty
    'engagement_signal': 0.2,    # 20% - user performance/engagement
}

# Data split parameters
TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.1
RANDOM_STATE = RANDOM_SEED
