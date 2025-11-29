import pandas as pd
import os
import shutil
import numpy as np
import sys
from datetime import datetime, timezone

BASE_DIR = "Input"
FAULT_FREE_DIR = os.path.join(BASE_DIR, "Fault-free")
FAULTY_DIR = os.path.join(BASE_DIR, "Faulty")
NUM_DAYS = 30
BASE_OUTPUT_DIR = "Output"

def read_data():
    fault_free_data = {}
    faulty_data = {}

    print(f"Starting data read from base directory: {BASE_DIR}")

    for day in range(1, NUM_DAYS + 1):
        filename = f"{day}_November.csv"

        ff_path = os.path.join(FAULT_FREE_DIR, filename)
        try:
            df_ff = pd.read_csv(ff_path)
            fault_free_data[day] = df_ff
        
        except FileNotFoundError:
            print(f"Warning: File not found at {ff_path}. Skipping.")
        except Exception as e:
            print(f"Error reading {ff_path}: {e}")

        f_path = os.path.join(FAULTY_DIR, filename)
        try:
            df_f = pd.read_csv(f_path)
            faulty_data[day] = df_f
        
        except FileNotFoundError:
            print(f"Warning: File not found at {f_path}. Skipping.")
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    print("-" * 30)
    print(f"Data read complete.")
    print(f"Loaded {len(fault_free_data)} fault-free files.")
    print(f"Loaded {len(faulty_data)} faulty files.")
    print("-" * 30)
    
    return fault_free_data, faulty_data

def align_clean_data(fault_free_data, faulty_data):
    aligned_ff_data = {}
    aligned_f_data = {}

    common_days = set(fault_free_data.keys()).intersection(set(faulty_data.keys()))

    print(f"Aligning and cleaning data for {len(common_days)} common days...")
    
    for day in sorted(list(common_days)):
        df_ff = fault_free_data[day]
        df_f = faulty_data[day]

        df_ff_clean = df_ff.dropna()
        df_f_clean = df_f.dropna()
        
        if df_ff_clean.empty or df_f_clean.empty:
            print(f"Warning: Day {day} has no data after 'dropna'. Skipping.")
            continue

        merged_df = pd.merge(
            df_ff_clean,
            df_f_clean,
            on='timestamp',
            how='inner',
            suffixes=('_ff', '_f')
        )
        
        if merged_df.empty:
            print(f"Warning: Day {day} has no common timestamps after cleaning. Skipping.")
            continue

        ff_cols = [col for col in df_ff.columns if col != 'timestamp']
        f_cols = [col for col in df_f.columns if col != 'timestamp']

        ff_final_cols_map = {'timestamp': 'timestamp'}
        for col in ff_cols:
            ff_final_cols_map[f"{col}_ff"] = col
        
        df_ff_final = merged_df[ff_final_cols_map.keys()].rename(
            columns=ff_final_cols_map
        )
        
        f_final_cols_map = {'timestamp': 'timestamp'}
        for col in f_cols:
            f_final_cols_map[f"{col}_f"] = col
            
        df_f_final = merged_df[f_final_cols_map.keys()].rename(
            columns=f_final_cols_map
        )

        aligned_ff_data[day] = df_ff_final
        aligned_f_data[day] = df_f_final

    print(f"Alignment and cleaning complete. {len(aligned_ff_data)} days processed.")
    print("-" * 30)
    
    return aligned_ff_data, aligned_f_data

GRANULARITIES = [2.5, 5, 10, 20, 25, 50]
SECONDS_IN_A_DAY = 24 * 60 * 60

def _segment_single_source(data_dict):
    segmented_data = {}

    for granularity in GRANULARITIES:
        granularity_str = f"{granularity}%"
        
        num_segments = 100 // granularity
        segment_duration_seconds = SECONDS_IN_A_DAY // num_segments
        
        print(f"  Processing granularity: {granularity_str} ({num_segments} segments, {segment_duration_seconds}s each)")

        segmented_data[granularity_str] = {}
        for i in range(int(num_segments)):
            tf_name = f"TF_{i}"
            segmented_data[granularity_str][tf_name] = {}
            
        for day, df in data_dict.items():
            if df.empty:
                continue
            try:
                day_start_dt = datetime(2020, 11, day, 0, 0, 0, tzinfo=timezone.utc)
                day_start_timestamp = int(day_start_dt.timestamp())
            except Exception as e:
                print(f"Warning: Could not create start timestamp for day {day}. Skipping. Error: {e}")
                continue
            
            for i in range(int(num_segments)):
                tf_name = f"TF_{i}"
                
                segment_start = day_start_timestamp + (i * segment_duration_seconds)
                segment_end = segment_start + segment_duration_seconds
                
                segment_df = df[
                    (df['timestamp'] >= segment_start) &
                    (df['timestamp'] < segment_end)
                ].copy()
                
                if not segment_df.empty:
                    segmented_data[granularity_str][tf_name][day] = segment_df

    return segmented_data

def segment_data(aligned_ff_data, aligned_f_data):
    
    print("--- Starting Data Segmentation ---")
    
    print("\nProcessing Fault-Free data...")
    segmented_ff_data = _segment_single_source(aligned_ff_data)
    
    print("\nProcessing Faulty data...")
    segmented_f_data = _segment_single_source(aligned_f_data)
    
    print("\n--- Data Segmentation Complete ---")
    print("-" * 30)
    
    return segmented_ff_data, segmented_f_data

def _save_single_source(segmented_data, base_path):
    
    for granularity_str, tf_dict in segmented_data.items():
        granularity_folder = granularity_str.replace('%', '')
        
        for tf_name, day_dict in tf_dict.items():
            
            for day, df in day_dict.items():
                if df.empty:
                    continue
                    
                final_dir = os.path.join(base_path, granularity_folder, tf_name)
                
                os.makedirs(final_dir, exist_ok=True)
                
                day_filename = f"{day}.csv"
                final_path = os.path.join(final_dir, day_filename)
                
                try:
                    print(f"  -> Saving: {final_path}")
                    
                    df.to_csv(final_path, index=False)
                except Exception as e:
                    print(f"Error: Could not write file to {final_path}. {e}")

def save_segmented_data(segmented_ff_data, segmented_f_data):
    
    print("--- Starting Data Saving ---")

    ff_output_path = os.path.join(BASE_OUTPUT_DIR, "Fault-free")
    f_output_path = os.path.join(BASE_OUTPUT_DIR, "Faulty")

    print(f"\nSaving Fault-Free data to {ff_output_path}...")
    _save_single_source(segmented_ff_data, ff_output_path)
    
    print(f"\nSaving Faulty data to {f_output_path}...")
    _save_single_source(segmented_f_data, f_output_path)
    
    print("\n--- Data Saving Complete ---")
    print("-" * 30)
		
fault_free_data, faulty_data = read_data()
fault_free_data, faulty_data = align_clean_data(fault_free_data, faulty_data)
fault_free_data, faulty_data = segment_data(fault_free_data, faulty_data)	
save_segmented_data(fault_free_data, faulty_data)