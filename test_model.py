import joblib
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import datetime
import sys
import time
import os
import warnings
import logging
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

def download_test_data_set():
    """Downloads test dataset if not present"""
    try:
        import opendatasets
        dataset_path = os.path.join("fraud-detection", "fraudTest.csv")
        if not os.path.exists(dataset_path):
            slow_print("Downloading test data set")
            opendatasets.download("https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTest.csv", force=True)
            return True
        return False
    except ImportError:
        logging.error("opendatasets module not found. Please install it using: pip install opendatasets")
        return False
    except Exception as e:
        logging.error(f"Error downloading dataset: {e}")
        return False

def select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select specific features from dataframe"""
    try:
        copy = df.copy()
        columns_to_drop = [col for col in df.columns if col not in features]
        copy = copy.drop(columns=columns_to_drop, axis=1)
        return copy
    except Exception as e:
        logging.error(f"Error selecting features: {e}")
        raise

def display_confusion_matrix(confusion_matrix: list[list[int]], non_fraud, fraud):
    """Display formatted confusion matrix with percentages"""
    try:
        nonfraud_count = len(non_fraud)
        fraud_count = len(fraud)
        
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        print("\nPredictions non-fraud  fraud")
        print("Actual")
        
        true_negative = confusion_matrix[0][0]
        false_positive = confusion_matrix[0][1]
        false_negative = confusion_matrix[1][0]
        true_positive = confusion_matrix[1][1]
        
        true_negative_rate = round(true_negative/nonfraud_count*100, 2)
        false_positive_rate = round(false_positive/nonfraud_count*100, 2)
        false_negative_rate = round(false_negative/fraud_count*100, 2)
        true_positive_rate = round(true_positive/fraud_count*100, 2)
        
        print(f"Non fraud   {true_negative_rate:>6}%   {false_positive_rate:>6}%")
        print(f"Fraud       {false_negative_rate:>6}%   {true_positive_rate:>6}%")
        
        # Additional metrics
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nAdditional Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        
    except Exception as e:
        logging.error(f"Error displaying confusion matrix: {e}")
        raise

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """Process and engineer features from raw dataframe"""
    try:
        new_df = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = ["Unnamed: 0", "cc_num", "city", "street", "state", "zip", "trans_num"]
        existing_columns_to_drop = [col for col in columns_to_drop if col in new_df.columns]
        new_df = new_df.drop(columns=existing_columns_to_drop, axis=1)
        
        # Generate age from DOB
        current_year = datetime.datetime.now().year
        new_df["age"] = new_df["dob"].apply(lambda x: current_year - int(x.split("-")[0]))
        new_df = new_df.drop(columns=["dob"], axis=1)
        
        # Clean merchant names (remove 'fraud_' prefix)
        new_df["merchant"] = new_df["merchant"].apply(lambda x: x.split("fraud_")[1] if "fraud_" in x else x)
        new_df = new_df.rename(columns={"merchant": "merch_name"})
        
        # Combine first and last name
        new_df["full_name"] = new_df["first"] + " " + new_df["last"]
        new_df = new_df.drop(columns=["first", "last"], axis=1)
        
        # Drop unix time if exists
        if "unix_time" in new_df.columns:
            new_df = new_df.drop(columns=["unix_time"], axis=1)
        
        # Extract datetime features
        datetime_format = "%Y-%m-%d %H:%M:%S"
        new_df["dow"] = new_df["trans_date_trans_time"].apply(
            lambda x: datetime.datetime.strptime(x, datetime_format).weekday()
        )
        new_df["dom"] = new_df["trans_date_trans_time"].apply(
            lambda x: int(x.split("-")[2].split(" ")[0])
        )
        new_df["month"] = new_df["trans_date_trans_time"].apply(
            lambda x: int(x.split("-")[1])
        )
        new_df["year"] = new_df["trans_date_trans_time"].apply(
            lambda x: int(x.split("-")[0])
        )
        
        # Process time into float
        def process_time(time: str) -> float:
            times = time.split(" ")[1].split(":")
            time_float = float(times[0]) + float(times[1])/60
            return time_float
        
        new_df["time"] = new_df["trans_date_trans_time"].apply(process_time)
        new_df = new_df.drop(columns=["trans_date_trans_time"], axis=1)
        
        # Create interval features
        new_df["time_interval"] = pd.cut(new_df["time"], bins=12).astype(str)
        new_df["amt_interval"] = pd.cut(new_df["amt"], bins=6).astype(str)
        new_df["age_interval"] = pd.cut(new_df["age"], bins=6).astype(str)
        new_df["city_pop_interval"] = pd.cut(new_df["city_pop"], bins=6).astype(str)
        
        # Reorder columns
        new_column_order = [
            "full_name", "gender", "age", "age_interval", "job", "lat", "long", 
            "city_pop", "city_pop_interval", "merch_name", "merch_lat", "merch_long", 
            "amt", "amt_interval", "category", "time", "time_interval", "dow", 
            "dom", "month", "year", "is_fraud"
        ]
        
        # Only reindex with columns that exist
        existing_columns = [col for col in new_column_order if col in new_df.columns]
        return new_df.reindex(columns=existing_columns)
        
    except Exception as e:
        logging.error(f"Error processing features: {e}")
        raise

def label_encode(df: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """Encode categorical features using LabelEncoder"""
    try:
        encoder = preprocessing.LabelEncoder()
        df_copy = df.copy()
        
        for feat in cat_features:
            if feat in df_copy.columns:
                df_copy[feat] = encoder.fit_transform(df_copy[feat])
            else:
                logging.warning(f"Feature '{feat}' not found in dataframe")
                
        return df_copy
    except Exception as e:
        logging.error(f"Error encoding features: {e}")
        raise

def slow_print(text, delay_time=0.01):
    """Print text character by character for visual effect"""
    for character in text:
        sys.stdout.write(character)
        sys.stdout.flush()
        time.sleep(delay_time)
    sys.stdout.write("\n")

def main():
    """Main execution function"""
    try:
        # Load model
        slow_print("Loading model")
        if not os.path.exists(config.MODEL_PATH):
            logging.error(f"{config.MODEL_PATH} not found in current directory")
            return
        model, features, target_class = joblib.load(config.MODEL_PATH)
        
        # Load dataset
        slow_print("Loading data set")
        data_set = None
        if os.path.exists(config.TEST_DATA_PATH):
            try:
                data_set = pd.read_csv(config.TEST_DATA_PATH)
                logging.info(f"Dataset loaded from: {config.TEST_DATA_PATH}")
            except Exception as e:
                logging.warning(f"Failed to load from {config.TEST_DATA_PATH}: {e}")
        
        if data_set is None:
            if download_test_data_set():
                data_set = pd.read_csv(os.path.join('fraud-detection', 'fraudTest.csv'))
            else:
                logging.error("Could not find or download the dataset")
                return
        
        # Separate features and target
        y = data_set[config.TARGET_COLUMN]
        x = data_set.drop(config.TARGET_COLUMN, axis=1)
        non_fraud = y[y == 0]
        fraud = y[y == 1]
        
        # Process features
        slow_print("Performing feature selection and extraction")
        x = process_features(x)
        
        slow_print("Performing label encoding")
        x = label_encode(x, config.CAT_FEATURES)
        
        slow_print("Performing final feature selection")
        x = select_features(x, features)
        
        # Make prediction for a specific instance
        instance = 69
        if instance < len(x):
            prediction = model.predict([x.iloc[instance]])[0]
            actual = y.iloc[instance]
            
            if prediction == 1 and prediction == actual:
                slow_print(f"Correctly flagged instance {instance} as FRAUD")
            elif prediction == 0 and prediction == actual:
                slow_print(f"Correctly flagged instance {instance} as NON-FRAUD")
            else:
                slow_print(f"Incorrect classification for instance {instance}")
                slow_print(f"Predicted: {'FRAUD' if prediction == 1 else 'NON-FRAUD'}")
                slow_print(f"Actual: {'FRAUD' if actual == 1 else 'NON-FRAUD'}")
        else:
            logging.warning(f"Instance {instance} is out of bounds")
        
        # Generate predictions for all data
        slow_print("Generating confusion matrix\n")
        all_predictions = model.predict(x)
        display_confusion_matrix(metrics.confusion_matrix(y, all_predictions), non_fraud, fraud)
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except ValueError as e:
        logging.error(f"Value error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()