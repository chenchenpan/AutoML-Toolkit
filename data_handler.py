import os
import pandas as pd
import numpy as np
import warnings
import json
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings("ignore")

class DataHandler:
    """Handles data loading and basic preprocessing operations."""
    
    def __init__(self):
        self.supported_extensions = {'.csv': ',', '.tsv': '\t'}
        self.missing_threshold = 0.6  # 60% threshold for missing values
    
    def read_file(self, path: str) -> Tuple[pd.DataFrame, str]:
        """
        Read a CSV or TSV file and return the dataframe and separator.
        
        Args:
            path: Path to the input file
            
        Returns:
            Tuple containing (DataFrame, separator character)
        """
        _, file_extension = os.path.splitext(path)
        if file_extension not in self.supported_extensions:
            raise ValueError(f'Unsupported file type: {file_extension}. Please use .csv or .tsv')
            
        sep = self.supported_extensions[file_extension]
        df = pd.read_csv(path, sep=sep)
        return df, sep

    def load_metadata(self, metadata_path: str) -> Dict:
        """
        Load metadata JSON file.
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            Dictionary containing metadata
        """
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata

    def fill_missing_values(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Fill missing values based on column types specified in metadata.
        
        Args:
            df: Input DataFrame
            metadata: Dictionary containing column type information
            
        Returns:
            DataFrame with filled missing values
        """
        # Validate all input features exist
        missing_cols = [col for col in metadata['input_features'] 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f'Input features {missing_cols} not found in the dataframe')

        # Fill missing values based on data type
        if metadata.get('input_int'):
            df[metadata['input_int']] = df[metadata['input_int']].fillna(0)
        if metadata.get('input_float'):
            df[metadata['input_float']] = df[metadata['input_float']].fillna(0.0)
        if metadata.get('input_categorical'):
            df[metadata['input_categorical']] = df[metadata['input_categorical']].fillna('UNK')
        if metadata.get('input_datetime'):
            # Convert datetime columns to datetime type if they aren't already
            for col in metadata['input_datetime']:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col])
                # Fill with minimum datetime that pandas can handle
                df[col] = df[col].fillna(pd.Timestamp.min)
        if metadata.get('input_bool'):
            df[metadata['input_bool']] = df[metadata['input_bool']].fillna(False)
        if metadata.get('input_text'):
            df[metadata['input_text']] = df[metadata['input_text']].fillna(' ')
            
        return df

    def drop_low_quality_columns(self, df: pd.DataFrame, missing_threshold: float = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Drop columns that have:
        1. More than threshold% missing values
        2. Only a single unique value (no variance)
        
        Args:
            df: Input DataFrame
            missing_threshold: Optional threshold for missing values (0.0-1.0)
                             Defaults to self.missing_threshold if not provided
        
        Returns:
            Tuple of (cleaned DataFrame, list of dropped column names)
        """
        if missing_threshold is None:
            missing_threshold = self.missing_threshold
            
        # Calculate missing value percentages
        missing_pcts = df.isnull().mean()
        high_missing_cols = missing_pcts[missing_pcts > missing_threshold].index.tolist()
        
        # Find columns with only one unique value
        single_value_cols = [col for col in df.columns 
                           if df[col].nunique() == 1]
        
        # Combine lists of columns to drop
        cols_to_drop = list(set(high_missing_cols + single_value_cols))
        
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} low quality columns:")
            print("\nColumns with > {:.0f}% missing values:".format(missing_threshold * 100))
            for col in high_missing_cols:
                print(f"- {col}: {missing_pcts[col]:.1%} missing")
            
            print("\nColumns with only one unique value:")
            for col in single_value_cols:
                print(f"- {col}: value = {df[col].iloc[0]}")
            
            df = df.drop(columns=cols_to_drop)
        
        return df, cols_to_drop

    def clean_data(self, df: pd.DataFrame, drop_cols: List[str], 
                  label: Optional[str] = None, 
                  auto_drop_low_quality: bool = True) -> pd.DataFrame:
        """
        Clean data by:
        1. Optionally dropping low quality columns
        2. Dropping specified columns
        3. Handling labels (optional)
        
        Args:
            df: Input DataFrame
            drop_cols: List of columns to drop
            label: Optional label handling:
                   - Column name to use as label
                   - 'has_label_signal' for signal-based labeling
                   - '0' or '1' for constant labeling
                   - None to skip label handling (default)
            auto_drop_low_quality: Whether to automatically drop low quality columns
            
        Returns:
            Cleaned DataFrame
        """
        if auto_drop_low_quality:
            df, dropped_cols = self.drop_low_quality_columns(df)
            # Add any auto-dropped columns to the manual drop list
            drop_cols = list(set(drop_cols + dropped_cols))
        
        # Handle labels only if label argument is provided
        if label is not None:
            # If label is 'has_label_signal':
            # - Creates a binary label column 'Label'
            # - Sets Label=1 if the signal column value is > 0
            # - Sets Label=0 if the signal column value is <= 0
            if label == 'has_label_signal':
                df['Label'] = np.where((df[label] > 0), 1, 0)
            # If label is '0' or '1':
            # - Creates a binary label column 'Label'
            # - Sets Label=1 if the label column value is '1'
            # - Sets Label=0 if the label column value is '0'
            elif label in ['0', '1']:
                df['Label'] = int(label)
            # If label is a column name:
            # - Fills missing values with 0
            elif label in df.columns:
                df[label].fillna(0, inplace=True)  # Fill missing values with 0
            else:
                raise ValueError(f'Unknown label type or column not found: {label}')
        
        new_df = df.drop(drop_cols, axis=1)
        return new_df

    def split_train_dev_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, dev, and test sets (80/10/10).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, dev_df, test_df)
        """
        shuffled_df = df.sample(frac=1)
        total = df.shape[0]
        num_test = int(total * 0.1)
        num_train = int(total * 0.8)
        
        train_df = shuffled_df.iloc[:num_train,:]
        dev_df = shuffled_df.iloc[num_train:num_train+num_test,:]
        test_df = shuffled_df.iloc[num_train+num_test:,:]
        
        return train_df, dev_df, test_df

    def save_data(self, df: pd.DataFrame, output_path: str, sep: str) -> None:
        """
        Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            output_path: Path where to save the file
            sep: Separator character to use
        """
        df.to_csv(output_path, sep=sep, index=False)

    def create_metadata(self, df: pd.DataFrame, output_dir: str, output_type: str, output_label: List[str]) -> Dict:
        """
        Create and save metadata based on the processed dataset.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save metadata.json
            output_type: Type of output - must be either 'classes', 'numbers', or a list of types for multi-task
            output_label: List of column names to be used as output labels
            
        Returns:
            Dictionary containing metadata
        """
        # Validate output_type
        if isinstance(output_type, str):
            if output_type not in ['classes', 'numbers']:
                raise ValueError("output_type must be either 'classes' or 'numbers'")
            output_types = [output_type] * len(output_label)
        elif isinstance(output_type, list):
            if not all(t in ['classes', 'numbers'] for t in output_type):
                raise ValueError("Each output_type must be either 'classes' or 'numbers'")
            if len(output_type) != len(output_label):
                raise ValueError("Length of output_type list must match length of output_label list")
            output_types = output_type
        else:
            raise ValueError("output_type must be either a string or a list")
        
        # Validate output_label
        if not isinstance(output_label, list) or not output_label:
            raise ValueError("output_label must be a non-empty list of column names")
        
        missing_cols = [col for col in output_label if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Output label columns {missing_cols} not found in the dataframe")

        def is_bool_column(series):
            """Helper function to identify boolean columns"""
            unique_values = set(series.dropna().unique())
            bool_values = {True, False, 1, 0, 'true', 'false', 'True', 'False', 'TRUE', 'FALSE', 'yes', 'no', 'Yes', 'No'}
            return unique_values.issubset(bool_values) and len(unique_values) <= 2

        def is_datetime_column(series):
            """Helper function to identify datetime columns"""
            if pd.api.types.is_datetime64_any_dtype(series):
                return True
            
            # Check if string values can be parsed as datetime
            if series.dtype == 'object':
                try:
                    # Try to parse a non-null sample value
                    sample = series.dropna().iloc[0] if not series.empty else None
                    if sample:
                        pd.to_datetime(sample)
                        # If successful, verify the whole column
                        pd.to_datetime(series, errors='raise')
                        return True
                except (ValueError, TypeError):
                    return False
            return False

        def is_categorical_column(series, threshold=0.05):
            """Helper function to identify categorical columns
            Args:
                series: pandas Series to check
                threshold: maximum ratio of unique values to total values to be considered categorical
            """
            if series.dtype == 'category':
                return True
            
            if series.dtype == 'object':
                # Calculate ratio of unique values to total values
                n_unique = series.nunique()
                n_total = len(series)
                unique_ratio = n_unique / n_total
                
                # Special handling for small datasets (less than 100 rows)
                if n_total < 100:
                    # For small datasets, primarily look at absolute number of unique values
                    if n_unique <= 5:  # If 5 or fewer unique values, consider it categorical
                        avg_length = series.str.len().mean()
                        return avg_length < 20      
                else:
                    # For larger datasets, use the ratio approach
                    if n_unique < 50:
                        avg_length = series.str.len().mean()
                        return (unique_ratio < threshold and 
                               avg_length < 20)
            return False

        # Identify column types based on dtype, excluding output label columns
        datetime_cols = [col for col in df.columns if is_datetime_column(df[col]) and col not in output_label]
        bool_cols = [col for col in df.columns if (df[col].dtype == 'bool' or 
                    (df[col].dtype == 'object' and is_bool_column(df[col]))) and col not in output_label]
        
        categorical_cols = [col for col in df.columns if 
                           (df[col].dtype == 'category' or 
                            (df[col].dtype == 'object' and 
                             is_categorical_column(df[col]))) and 
                           col not in output_label and 
                           col not in bool_cols and 
                           col not in datetime_cols]
        
        # Then remaining object columns are text
        text_cols = [col for col in df.columns if 
                    df[col].dtype == 'object' and 
                    col not in categorical_cols and 
                    col not in bool_cols and 
                    col not in datetime_cols and 
                    col not in output_label]
        
        float_cols = [col for col in df.columns if df[col].dtype == 'float64' and col not in output_label]
        int_cols = [col for col in df.columns if df[col].dtype == 'int64' and col not in output_label]
        
        metadata = {
            'output_type': output_type,
            'output_types': output_types,
            'input_features': [col for col in df.columns if col not in output_label],
            'output_label': output_label,
            'input_text': text_cols,
            'input_float': float_cols,
            'input_int': int_cols,
            'input_categorical': categorical_cols,
            'input_datetime': datetime_cols,
            'input_bool': bool_cols
        }
        
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return metadata

    def analyze_dataset(self, df: pd.DataFrame) -> None:
        """
        Analyze dataset and print detailed information about features and statistics.
        
        Args:
            df: Input DataFrame
        """
        # Dataset Overview
        print("\n=== Dataset Overview ===")
        print(f"Number of samples: {len(df):,}")
        print(f"Number of features: {len(df.columns):,}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Feature Types Summary
        print("\n=== Feature Types Summary ===")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count} columns")
        
        # Detailed Feature Analysis
        print("\n=== Detailed Feature Analysis ===")
        for column in df.columns:
            print(f"\nColumn: {column}")
            print(f"Type: {df[column].dtype}")
            print(f"Unique values: {df[column].nunique():,}")
            print(f"Missing values: {df[column].isnull().sum():,} ({df[column].isnull().mean():.2%})")
            
            # Show sample values based on dtype
            if df[column].dtype == 'object':
                sample_values = df[column].dropna().sample(min(3, df[column].nunique()))
                print("Sample values:", sample_values.tolist())
            elif pd.api.types.is_numeric_dtype(df[column]):
                print(f"Min: {df[column].min()}")
                print(f"Max: {df[column].max()}")
                print(f"Mean: {df[column].mean():.2f}")
            
            # Memory usage
            memory_usage = df[column].memory_usage() / 1024**2
            print(f"Memory usage: {memory_usage:.2f} MB")
        
        # Missing Values Summary
        missing_data = df.isnull().sum()[df.isnull().sum() > 0]
        if not missing_data.empty:
            print("\n=== Missing Values Summary ===")
            for column, count in missing_data.items():
                print(f"{column}: {count:,} missing values ({count/len(df):.2%})")
        
        # Correlation Analysis for Numeric Columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            print("\n=== High Correlations (>0.7) ===")
            corr_matrix = df[numeric_cols].corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.7)
            high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                         for x, y in zip(*high_corr) if x != y and x < y]
            
            for col1, col2, corr in high_corr:
                print(f"{col1} - {col2}: {corr:.2f}")