# config.py

# Data paths
DATA_PATH = r'C:\Users\1012 G2\Documents\archive\fraudTrain.csv'
MODEL_PATH = 'model.pkl'
TEST_DATA_PATH = r'C:\Users\1012 G2\Documents\archive\fraudTest.csv'

# Feature engineering
DROP_COLUMNS = ["Unnamed: 0", "cc_num", "city", "street", "state", "zip", "trans_num"]
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Categorical features for encoding
CAT_FEATURES = [
    "full_name", "gender", "job", "merch_name", "category", "time_interval", "age_interval", "amt_interval", "city_pop_interval", "distance_interval"
]

# Model features
MODEL_FEATURES = ["amt", "category", "time", "city_pop", "age", "gender", "month"]

# Target column
TARGET_COLUMN = "is_fraud"

# Model parameters
RANDOM_STATE = 69

# Train/test split
TEST_SIZE = 0.2

# Progress bar
PROGRESS_BAR_COLOR = 'green'
PROGRESS_BAR_NCOLS = 80
