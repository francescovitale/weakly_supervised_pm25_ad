import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import joblib
import shutil
import copy
import warnings
import sys
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, TimeoutError

INPUT_DIR = "Input/"
ARPA_FILE_PATH = os.path.join(INPUT_DIR, "arpa.csv") 
OUTPUT_DIR = "Output/" # Main output directory

MODELS_DIR = os.path.join(OUTPUT_DIR, "calibration_models")
CALIBRATED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "calibrated_output") # We'll save metrics AND data here
METRICS_FILE = os.path.join(CALIBRATED_OUTPUT_DIR, "calibration_metrics.csv")
AGGREGATED_METRICS_FILE = os.path.join(CALIBRATED_OUTPUT_DIR, "aggregated_metrics.csv")


GRANULARITIES = {
	"0.1": 999,
	"1": 100,
	"2": 50,
	"2.5": 40,
	"5": 20,
	"10": 10,
	"20": 5,
	"25": 4,
	"50": 2
}
SECONDS_IN_A_DAY = 86400

ARPA_COL = 'pm25' 
AGGREGATED_COL = 'pm25_fa' 
SENSOR_COLS = ["pm25_0", "pm25_1", "pm25_2", "pm25_3"]

def load_segmented_data():
	print(f"Loading segmented sensor data from '{INPUT_DIR}'...")
	fault_free_data = {}
	faulty_data = {}
	
	data_sources = {'Fault-free': fault_free_data, 'Faulty': faulty_data}
	
	for source_type, data_dict in data_sources.items():
		source_path = os.path.join(INPUT_DIR, source_type)
		
		if not os.path.isdir(source_path):
			print(f"Warning: Directory not found, skipping: {source_path}")
			continue
			
		for gran_str in os.listdir(source_path):
			if gran_str not in GRANULARITIES:
				continue
			
			gran_path = os.path.join(source_path, gran_str)
			if not os.path.isdir(gran_path):
				continue
				
			data_dict[gran_str] = {}
			
			for tf_name in os.listdir(gran_path):
				if not tf_name.startswith('TF_'):
					continue
					
				tf_path = os.path.join(gran_path, tf_name)
				if not os.path.isdir(tf_path):
					continue
					
				data_dict[gran_str][tf_name] = {}
				
				for day_file in os.listdir(tf_path):
					if not day_file.endswith('.csv'):
						continue
					
					day_str = os.path.splitext(day_file)[0]
					file_path = os.path.join(tf_path, day_file)
					
					try:
						df = pd.read_csv(file_path)
						if 'timestamp' in df.columns:
							df['timestamp'] = df['timestamp'].astype(int)
						data_dict[gran_str][tf_name][day_str] = df
					except Exception as e:
						print(f"Warning: Could not read {file_path}. Error: {e}")
						
	print("...Sensor data loading complete.")
	return fault_free_data, faulty_data

def load_arpa_data(fault_free_data, faulty_data, arpa_col='pm25'):
	import pandas as pd

	print(f"Loading ARPA data from '{ARPA_FILE_PATH}'...")

	def _find_global_time_window(segmented_data_list):
		all_timestamps = []
		for data_dict in segmented_data_list:
			for granularity, time_frames in data_dict.items():
				for tf_name, days in time_frames.items():
					for day, df in days.items():
						if isinstance(df, pd.DataFrame) and not df.empty:
							all_timestamps.extend(df['timestamp'].values)
		if not all_timestamps:
			return None, None
		return int(min(all_timestamps)), int(max(all_timestamps))

	min_ts, max_ts = _find_global_time_window([fault_free_data, faulty_data])
	
	
	if min_ts is None:
		print("Warning: No sensor data loaded. Cannot determine time window for ARPA.")
		return pd.DataFrame()

	print(f"  > Found sensor data time window: {min_ts} to {max_ts}")

	try:
		arpa_df = pd.read_csv(ARPA_FILE_PATH)

		if 'timestamp' not in arpa_df.columns:
			print(f"Error: 'timestamp' column not found in {ARPA_FILE_PATH}")
			return pd.DataFrame()
		if arpa_col not in arpa_df.columns:
			print(f"Error: '{arpa_col}' column not found in {ARPA_FILE_PATH}")
			return pd.DataFrame()
		arpa_df['datetime'] = pd.to_datetime(arpa_df['timestamp'])
		arpa_df = arpa_df.dropna(subset=['datetime'])

		arpa_df['datetime'] = arpa_df['datetime'].dt.tz_localize('Etc/GMT-1')

		arpa_df['datetime_utc'] = arpa_df['datetime'].dt.tz_convert('UTC')

		arpa_df['unix_timestamp'] = arpa_df['datetime_utc'].astype('int64') // 10**9

	except FileNotFoundError:
		print(f"Error: ARPA file not found at {ARPA_FILE_PATH}")
		return pd.DataFrame()
	except Exception as e:
		print(f"Error reading {ARPA_FILE_PATH}: {e}")
		return pd.DataFrame()

	original_rows = len(arpa_df)
	# Widen the filter slightly just in case of interpolation edge effects
	padding = 3600 # 1 hour
	arpa_filtered = arpa_df[
		(arpa_df['unix_timestamp'] >= min_ts - padding) & 
		(arpa_df['unix_timestamp'] <= max_ts + padding)
	].copy()
	
	filtered_rows = len(arpa_filtered)

	print(f"  > Filtered ARPA data: kept {filtered_rows} of {original_rows} rows.")
	if filtered_rows == 0:
		print(f"Warning: No ARPA data found within the sensor time window.")

	return arpa_filtered[['datetime', arpa_col, 'unix_timestamp']]
	
def interpolate_arpa_data(arpa_filtered_df, freq='1S', method='time', smooth_window=None):
	import pandas as pd
	import numpy as np

	if arpa_filtered_df.empty:
		print("Warning: Cannot interpolate empty ARPA DataFrame.")
		return pd.DataFrame()

	arpa_indexed = arpa_filtered_df.set_index('datetime').copy()

	arpa_resampled = arpa_indexed.resample(freq)

	if method in ['time', 'linear', 'nearest']:
		arpa_interpolated = arpa_resampled.interpolate(method=method)
	elif method in ['spline', 'polynomial']:
		arpa_interpolated = arpa_resampled.interpolate(method=method, order=3)
	elif method in ['ffill', 'bfill']:
		arpa_interpolated = getattr(arpa_resampled, method)()
	else:
		raise ValueError(f"Unsupported interpolation method: {method}")

	arpa_interpolated = arpa_interpolated.ffill().bfill()

	if smooth_window is not None:
		arpa_interpolated = arpa_interpolated.rolling(smooth_window, min_periods=1).mean()

	arpa_final = arpa_interpolated.reset_index()
	arpa_final['timestamp'] = arpa_final['datetime'].astype('int64') // 10**9

	print(f"...Interpolation complete. {len(arpa_final)} records created using method='{method}'"
		  f"{', smoothed' if smooth_window else ''}.")
	return arpa_final.drop(columns=['datetime'])

def align_data_for_training(segmented_data, arpa_high_freq_df):

	print("Aligning sensor data with interpolated ARPA data...")
	aligned_data = {}
	
	if arpa_high_freq_df.empty:
		print("Warning: ARPA data is empty, cannot align. Returning empty dict.")
		return aligned_data

	for gran_str, time_frames in segmented_data.items():
		aligned_data[gran_str] = {}
		for tf_name, days in time_frames.items():
			aligned_data[gran_str][tf_name] = {}
			for day_str, sensor_df in days.items():
				
				if sensor_df.empty:
					print(f"  > Skipping empty segment: {gran_str}/{tf_name}/{day_str}")
					aligned_data[gran_str][tf_name][day_str] = pd.DataFrame()
					continue

				merged_df = pd.merge(
					sensor_df,
					arpa_high_freq_df,
					on='timestamp',
					how='inner'
				)
				
				merged_df = merged_df.dropna()
				aligned_data[gran_str][tf_name][day_str] = merged_df

	print("...Data alignment complete.")
	return aligned_data

def apply_filter_and_aggregation(aligned_data_dict, is_faulty=False, ema_span=60):
	procedure_name = "full F+A (EMA)" if is_faulty else "simple aggregation"
	print(f"Applying {procedure_name}...")

	fa_data = copy.deepcopy(aligned_data_dict)

	floor_threshold = 5.0
	consistency_threshold = 10

	for gran_str, time_frames in fa_data.items():
		for tf_name, days in time_frames.items():
			for day_str, df in days.items():

				if df.empty:
					continue

				if is_faulty:
					fa_df = df[SENSOR_COLS].copy()

					for col in SENSOR_COLS:
						fa_df[col] = fa_df[col].ewm(span=ema_span, adjust=False).mean()

					fa_df = fa_df.where(fa_df >= floor_threshold, np.nan)

					row_median = fa_df[SENSOR_COLS].median(axis=1, skipna=True)
					deviations = fa_df[SENSOR_COLS].subtract(row_median, axis=0).abs()
					row_mad = deviations.median(axis=1, skipna=True) + 1.0
					consistency_score = deviations.divide(row_mad, axis=0)
					fa_df = fa_df.where(consistency_score <= consistency_threshold, np.nan)

				else:
					fa_df = df[SENSOR_COLS].copy()

				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=RuntimeWarning)
					aggregated_series = fa_df[SENSOR_COLS].mean(axis=1, skipna=True)

				fa_data[gran_str][tf_name][day_str][AGGREGATED_COL] = aggregated_series
				fa_data[gran_str][tf_name][day_str] = (
					fa_data[gran_str][tf_name][day_str].dropna(subset=[AGGREGATED_COL])
				)

	print(f"...{procedure_name} complete.")
	return fa_data

# ---
# [NEW FUNCTION]
# ---
def train_global_calibration_model(aligned_ff_fa_data, method='linear', knn_params=None, svr_params=None, linear_svr=False, xgb_params=None):
	"""
	Trains ONE global model on ALL available fault-free F+A data.
	"""
	print(f"Training GLOBAL calibration model using method='{method}'...")
	
	dfs = []
	# Iterate through all segments and collect all available fault-free data
	for gran_str, time_frames in aligned_ff_fa_data.items():
		for tf_name, days in time_frames.items():
			for day_str, df in days.items():
				if not df.empty and AGGREGATED_COL in df.columns:
					dfs.append(df)

	if not dfs:
		print("  > Error: No F+A data found anywhere. Cannot train global model.")
		return None

	all_data_df = pd.concat(dfs, ignore_index=True)

	if all_data_df.empty or all_data_df.shape[0] < 2:
		print(f"  > Error: Not enough data ({all_data_df.shape[0]} rows) to train global model.")
		return None

	print(f"  > Training global model on {all_data_df.shape[0]} total data points.")

	X = all_data_df[[AGGREGATED_COL]]
	y = all_data_df[ARPA_COL]

	if method.lower() == 'linear':
		model = LinearRegression()
		model.fit(X, y)

	elif method.lower() == 'svr':
		if linear_svr:
			svr_params = svr_params or {'C': 1.0, 'epsilon': 0.1, 'max_iter': 5000, 'dual': False}
			model = LinearSVR(**svr_params)
		else:
			svr_params = svr_params or {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}
			model = SVR(**svr_params)
		model.fit(X, y)

	elif method.lower() == 'xgboost':
		xgb_params = xgb_params or {}
		model = xgb.XGBRegressor(**xgb_params)
		model.fit(X, y)

	elif method.lower() == 'knn':
		knn_params = knn_params or {'n_neighbors': 5, 'weights': 'uniform'}
		model = KNeighborsRegressor(**knn_params)
		model.fit(X, y)

	else:
		raise ValueError(f"Unknown method '{method}'. Choose 'linear', 'svr', 'xgboost', or 'knn'.")

	print("...Global model training complete.")
	return model


def train_individual_sensor_models(aligned_f_data, method='linear', knn_params=None, svr_params=None, linear_svr=False, xgb_params=None):
	print(f"Training individual sensor models using method='{method}'...")
	models = {}

	for gran_str, time_frames in aligned_f_data.items():
		for tf_name, days in time_frames.items():
			dfs = []
			for day_str, df in days.items():
				if not df.empty and ARPA_COL in df.columns:
					dfs.append(df)

			if not dfs:
				print(f"  > Warning: No training data (with {ARPA_COL}) found for {gran_str}/{tf_name}. Skipping.")
				continue

			all_days_df = pd.concat(dfs, ignore_index=True)

			for col in SENSOR_COLS:
				if col not in all_days_df.columns:
					print(f"  > Info: Sensor {col} not in data for {gran_str}/{tf_name}. Skipping.")
					continue
				
				training_data = all_days_df[[col, ARPA_COL]].dropna()
				
				if training_data.empty or training_data.shape[0] < 2:
					print(f"  > Warning: Not enough data for {col} in {gran_str}/{tf_name}.")
					continue

				X = training_data[[col]]
				y = training_data[ARPA_COL]

				if method.lower() == 'linear':
					model = LinearRegression()
					model.fit(X, y)

				elif method.lower() == 'svr':
					if linear_svr:
						svr_params = svr_params or {'C': 1.0, 'epsilon': 0.1, 'max_iter': 5000, 'dual': False}
						model = LinearSVR(**svr_params)
					else:
						svr_params = svr_params or {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}
						model = SVR(**svr_params)
					model.fit(X, y)

				elif method.lower() == 'xgboost':
					xgb_params = xgb_params or {}
					model = xgb.XGBRegressor(**xgb_params)
					model.fit(X, y)

				elif method.lower() == 'knn':
					knn_params = knn_params or {'n_neighbors': 5, 'weights': 'uniform'}
					model = KNeighborsRegressor(**knn_params)
					model.fit(X, y)

				else:
					raise ValueError(f"Unknown method '{method}'.")

				models[(gran_str, tf_name, col)] = model 

	print("...Individual sensor model training complete.")
	return models

# ---
# [NEW FUNCTION]
# ---
def save_global_model(model, base_dir, model_name):
	"""
	Saves a single global model object.
	"""
	if model is None:
		print("Warning: Model is None. Skipping saving.")
		return
		
	print(f"Saving global model to '{base_dir}'...")
	os.makedirs(base_dir, exist_ok=True)
	model_filename = f"{model_name}.pkl"
	model_path = os.path.join(base_dir, model_filename)
	
	try:
		joblib.dump(model, model_path)
		print(f"...Global model saved to {model_path}.")
	except Exception as e:
		print(f"Warning: Could not save global model {model_path}. Error: {e}")


def safe_predict(model, X, timeout_sec=60.0):
	with ThreadPoolExecutor(max_workers=1) as executor:
		future = executor.submit(model.predict, X)
		try:
			result = future.result(timeout=timeout_sec)
			return result
		except TimeoutError:
			print(f"  > Warning: Prediction exceeded {timeout_sec}s — using NaN values.")
			return np.full(len(X), np.nan)
		except Exception as e:
			print(f"  > Error during prediction: {e}")
			return np.full(len(X), np.nan)

# ---
# [MODIFIED FUNCTION]
# ---
def apply_calibration_to_aggregate(global_model, aligned_data, collection_name="", timeout_sec=60.0):
	"""
	Applies the one global_model to all aggregate data segments.
	"""
	print(f"Applying global model '{collection_name}' to aggregate data...")

	if not global_model:
		 print("  > Error: global_model is None. Cannot apply calibration.")
		 return

	for gran_str, time_frames in aligned_data.items():
		for tf_name, days in time_frames.items():
			# No model lookup needed, we use the global_model
			for day_str, df in days.items():
				if df.empty or AGGREGATED_COL not in df.columns:
					continue

				valid_rows = df[[AGGREGATED_COL]].dropna()
				if not valid_rows.empty:
					pred_vals = safe_predict(global_model, valid_rows, timeout_sec)
					df.loc[valid_rows.index, f'cal_{AGGREGATED_COL}'] = pred_vals

	print("...Aggregate calibration complete.")


def apply_sensor_specific_calibration(models, aligned_f_data, collection_name="", timeout_sec=60.0):
	print(f"Applying sensor-specific models from '{collection_name}'...")

	for gran_str, time_frames in aligned_f_data.items():
		for tf_name, days in time_frames.items():
			for day_str, df in days.items():
				if df.empty:
					continue

				for col in SENSOR_COLS:
					if col not in df.columns:
						continue

					model = models.get((gran_str, tf_name, col))
					if not model:
						continue

					valid_rows = df[[col]].dropna()
					if not valid_rows.empty:
						pred_vals = safe_predict(model, valid_rows, timeout_sec)
						df.loc[valid_rows.index, f'cal_{col}'] = pred_vals

	print("...Sensor-specific calibration complete.")

# ---
# [MODIFIED FUNCTION]
# ---
def apply_aggregate_model_to_individual_sensors(global_model, aligned_raw_data, collection_name="", timeout_sec=60.0):
	"""
	Applies the one global_model to all individual raw sensor columns.
	"""
	print(f"Applying global model '{collection_name}' to individual raw sensor columns...")

	if not global_model:
		 print("  > Error: global_model is None. Cannot apply calibration.")
		 return

	for gran_str, time_frames in aligned_raw_data.items():
		for tf_name, days in time_frames.items():
			# No model lookup needed, we use the global_model
			for day_str, df in days.items():
				if df.empty:
					continue

				# Now, iterate over each sensor column and apply the SAME global model
				for col in SENSOR_COLS:
					if col not in df.columns:
						continue

					valid_rows = df[[col]].dropna()
					if not valid_rows.empty:
						# [FIX] Rename the column to match the model's expected input feature name ('pm25_fa')
						valid_rows_renamed = valid_rows.rename(columns={col: AGGREGATED_COL})
						
						# Apply the global_model to the individual sensor column 'col'
						pred_vals = safe_predict(global_model, valid_rows_renamed, timeout_sec)
						df.loc[valid_rows.index, f'cal_{col}'] = pred_vals

	print("...Individual sensor calibration (using global model) complete.")


def save_calibrated_data(aligned_ff_fa_data, aligned_f_fa_data, aligned_f_data, base_output_dir):
	print(f"Saving calibrated data windows to '{base_output_dir}'...")

	all_grans = set(aligned_ff_fa_data.keys()) | set(aligned_f_fa_data.keys()) | set(aligned_f_data.keys())

	for gran_str in all_grans:
		all_tfs = set(aligned_ff_fa_data.get(gran_str, {}).keys()) | \
				  set(aligned_f_fa_data.get(gran_str, {}).keys()) | \
				  set(aligned_f_data.get(gran_str, {}).keys())
		
		for tf_name in all_tfs:
			output_path = os.path.join(base_output_dir, gran_str, tf_name)
			os.makedirs(output_path, exist_ok=True)

			all_days = set(aligned_ff_fa_data.get(gran_str, {}).get(tf_name, {}).keys()) | \
					   set(aligned_f_fa_data.get(gran_str, {}).get(tf_name, {}).keys()) | \
					   set(aligned_f_data.get(gran_str, {}).get(tf_name, {}).keys())
			
			for day_str in all_days:
				
				if day_str in aligned_ff_fa_data.get(gran_str, {}).get(tf_name, {}):
					ff_fa_df = aligned_ff_fa_data[gran_str][tf_name][day_str]
					if not ff_fa_df.empty:
						cols_to_save = ['timestamp', ARPA_COL, AGGREGATED_COL, f'cal_{AGGREGATED_COL}']
						final_cols = [col for col in cols_to_save if col in ff_fa_df.columns]
						if final_cols:
							filename = f"{day_str}_fault_free_fa.csv"
							ff_fa_df[final_cols].to_csv(os.path.join(output_path, filename), index=False)

				if day_str in aligned_f_fa_data.get(gran_str, {}).get(tf_name, {}):
					f_fa_df = aligned_f_fa_data[gran_str][tf_name][day_str]
					if not f_fa_df.empty:
						cols_to_save = ['timestamp', ARPA_COL, AGGREGATED_COL, f'cal_{AGGREGATED_COL}']
						final_cols = [col for col in cols_to_save if col in f_fa_df.columns]
						if final_cols:
							filename = f"{day_str}_faulty_fa.csv"
							f_fa_df[final_cols].to_csv(os.path.join(output_path, filename), index=False)

				if day_str in aligned_f_data.get(gran_str, {}).get(tf_name, {}):
					f_raw_df = aligned_f_data[gran_str][tf_name][day_str]
					if not f_raw_df.empty:
						cal_cols = [f'cal_{col}' for col in SENSOR_COLS]
						cols_to_save = ['timestamp', ARPA_COL] + SENSOR_COLS + cal_cols
						final_cols = [col for col in cols_to_save if col in f_raw_df.columns]
						if final_cols:
							filename = f"{day_str}_faulty_raw.csv"
							f_raw_df[final_cols].to_csv(os.path.join(output_path, filename), index=False)

	print("...Calibrated data window saving complete.")

def calculate_and_save_metrics(aligned_ff_fa_data, aligned_f_fa_data, aligned_f_data, output_file):
	print(f"Calculating and saving metrics (per-timeframe) to {output_file}...")
	results = []
	
	all_grans = set(aligned_ff_fa_data.keys()) | set(aligned_f_fa_data.keys()) | set(aligned_f_data.keys())

	for gran_str in all_grans:
		all_tfs = set(aligned_ff_fa_data.get(gran_str, {}).keys()) | \
				  set(aligned_f_fa_data.get(gran_str, {}).keys()) | \
				  set(aligned_f_data.get(gran_str, {}).keys())
		
		for tf_name in all_tfs:
			all_days_in_tf = set(aligned_ff_fa_data.get(gran_str, {}).get(tf_name, {}).keys()) | \
							 set(aligned_f_fa_data.get(gran_str, {}).get(tf_name, {}).keys()) | \
							 set(aligned_f_data.get(gran_str, {}).get(tf_name, {}).keys())

			dfs_ff_fa = []
			for day_str in all_days_in_tf:
				if day_str in aligned_ff_fa_data.get(gran_str, {}).get(tf_name, {}):
					dfs_ff_fa.append(aligned_ff_fa_data[gran_str][tf_name][day_str])
			
			if dfs_ff_fa:
				all_days_ff = pd.concat(dfs_ff_fa, ignore_index=True)
				cal_col = f'cal_{AGGREGATED_COL}'
				valid_rows = all_days_ff[[ARPA_COL, cal_col]].dropna()
				
				if len(valid_rows) >= 2:
					y_true = valid_rows[ARPA_COL]
					y_pred = valid_rows[cal_col]
					r2 = r2_score(y_true, y_pred)
					rmse = np.sqrt(mean_squared_error(y_true, y_pred))
					pearson_r, _ = pearsonr(y_true, y_pred)
					
					results.append({
						'granularity': gran_str, 'time_frame': tf_name,
						'data_type': 'fault_free_fa', 'sensor': AGGREGATED_COL,
						'r2': r2, 'rmse': rmse, 'pearson_r': pearson_r,
						'n_points': len(valid_rows)
					})
			
			dfs_f_fa = []
			for day_str in all_days_in_tf:
				if day_str in aligned_f_fa_data.get(gran_str, {}).get(tf_name, {}):
					dfs_f_fa.append(aligned_f_fa_data[gran_str][tf_name][day_str])
			
			if dfs_f_fa:
				all_days_f = pd.concat(dfs_f_fa, ignore_index=True)
				cal_col = f'cal_{AGGREGATED_COL}'
				valid_rows = all_days_f[[ARPA_COL, cal_col]].dropna()

				if len(valid_rows) >= 2:
					y_true = valid_rows[ARPA_COL]
					y_pred = valid_rows[cal_col]
					r2 = r2_score(y_true, y_pred)
					rmse = np.sqrt(mean_squared_error(y_true, y_pred))
					pearson_r, _ = pearsonr(y_true, y_pred)
					
					results.append({
						'granularity': gran_str, 'time_frame': tf_name,
						'data_type': 'faulty_fa', 'sensor': AGGREGATED_COL,
						'r2': r2, 'rmse': rmse, 'pearson_r': pearson_r,
						'n_points': len(valid_rows)
					})

			dfs_f_raw = []
			for day_str in all_days_in_tf:
				if day_str in aligned_f_data.get(gran_str, {}).get(tf_name, {}):
					dfs_f_raw.append(aligned_f_data[gran_str][tf_name][day_str])
			
			if dfs_f_raw:
				all_days_raw = pd.concat(dfs_f_raw, ignore_index=True)
				if not all_days_raw.empty:
					for col in SENSOR_COLS:
						cal_col = f'cal_{col}'
						valid_rows = all_days_raw[[ARPA_COL, cal_col]].dropna()
						
						if len(valid_rows) >= 2:
							y_true = valid_rows[ARPA_COL]
							y_pred = valid_rows[cal_col]
							r2 = r2_score(y_true, y_pred)
							rmse = np.sqrt(mean_squared_error(y_true, y_pred))
							pearson_r, _ = pearsonr(y_true, y_pred)
							
							results.append({
								'granularity': gran_str, 'time_frame': tf_name,
								'data_type': 'faulty_raw', 'sensor': col,
								'r2': r2, 'rmse': rmse, 'pearson_r': pearson_r,
								'n_points': len(valid_rows)
							})
								
	if results:
		results_df = pd.DataFrame(results)
		os.makedirs(os.path.dirname(output_file), exist_ok=True)
		results_df.to_csv(output_file, index=False)
		print(f"...Metrics saved. {len(results_df)} per-timeframe results calculated.")
	else:
		print("...No metrics were calculated.")

def aggregate_and_save_final_results(input_file, output_file):
	print(f"Aggregating final results from {input_file}...")
	try:
		df = pd.read_csv(input_file, dtype={'granularity': str})
	except FileNotFoundError:
		print(f"Warning: Metrics file not found at {input_file}. Cannot aggregate.")
		return
	except Exception as e:
		print(f"Error reading {input_file}: {e}")
		return
		
	if df.empty:
		print(" > Warning: Metrics file is empty. Nothing to aggregate.")
		return

	agg_df = df.groupby(['granularity', 'data_type']).agg(
		r2_mean=('r2', 'mean'),
		r2_std=('r2', 'std'),
		rmse_mean=('rmse', 'mean'),
		rmse_std=('rmse', 'std'),
		pearson_r_mean=('pearson_r', 'mean'), 
		pearson_r_std=('pearson_r', 'std')	
	).reset_index()
	
	print(agg_df)

	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	agg_df.to_csv(output_file, index=False, float_format='%.4f')
	print(f"...Aggregated metrics (mean + std) saved to {output_file}")

# ---
# [MODIFIED FUNCTION]
# ---
def main():

	try:
		method = sys.argv[1]
	except:
		print("Usage: python your_script_name.py <method_name>")
		print("Available methods: linear_regression, svr, linear_svr, xgboost, knn")
		sys.exit()
	
	print("=== CALIBRATION SYSTEM START ===")
	# 1. load the segmented data
	fault_free_data, faulty_data = load_segmented_data()
	
	# 2. load the ARPA data and interpolate it across the whole month
	arpa_df = load_arpa_data(fault_free_data, faulty_data)
	arpa_interpolated = interpolate_arpa_data(arpa_df, freq='1S', method='time')

	# Align raw data
	aligned_ff = align_data_for_training(fault_free_data, arpa_interpolated) # Used for F+A
	aligned_f = align_data_for_training(faulty_data, arpa_interpolated)     # Used for F+A and raw application

	# Create F+A datasets
	aligned_ff_fa = apply_filter_and_aggregation(aligned_ff, is_faulty=False) # Used for global training
	aligned_f_fa = apply_filter_and_aggregation(aligned_f, is_faulty=True)  # Used for aggregate application
	
	# 3. train a calibration model against the fault-free data
	print("\n[Training ONE GLOBAL model on ALL FAULT-FREE data]...")
	
	global_model = None
	model_params = {}
	model_name = f"global_{method}_model"
	
	if method == "linear_regression":
		global_model = train_global_calibration_model(aligned_ff_fa, method='linear')
	
	elif method == "svr":
		model_params = {'method': 'svr', 'svr_params': {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}}
		global_model = train_global_calibration_model(aligned_ff_fa, **model_params)
	
	elif method == "linear_svr":
		model_params = {'method': 'svr', 'linear_svr': True, 'svr_params': {'C': 1.0, 'epsilon': 0.1, 'max_iter': 1000}}
		global_model = train_global_calibration_model(aligned_ff_fa, **model_params)
	
	elif method == "xgboost":
		model_params = {'method': 'xgboost', 'xgb_params': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}}
		global_model = train_global_calibration_model(aligned_ff_fa, **model_params)
	
	elif method == "knn":
		model_params = {'method': 'knn', 'knn_params': {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'kd_tree'}}
		global_model = train_global_calibration_model(aligned_ff_fa, **model_params)
	else:
		print(f"Error: Unknown method '{method}'.")
		sys.exit()

	# Save the single global model
	save_global_model(global_model, MODELS_DIR, model_name)
	
	# 4. calibrate the windows of fault-free, faulty_fa, and faulty_raw data with the global model
	
	# Apply GLOBAL model to fault-free F+A data
	print("\n[Applying GLOBAL model to fault-free F+A data...]")
	apply_calibration_to_aggregate(global_model, aligned_ff_fa, model_name)
	
	# Apply GLOBAL model to faulty F+A data
	print("\n[Applying GLOBAL model to faulty F+A data...]")
	apply_calibration_to_aggregate(global_model, aligned_f_fa, model_name)
	
	# Apply GLOBAL model to raw faulty sensor data (aligned_f)
	print("\n[Applying GLOBAL model to raw faulty sensor data...]")
	apply_aggregate_model_to_individual_sensors(global_model, aligned_f, model_name)
	
	# 5. do the rest.
	save_calibrated_data(aligned_ff_fa, aligned_f_fa, aligned_f, CALIBRATED_OUTPUT_DIR)
	
	calculate_and_save_metrics(aligned_ff_fa, aligned_f_fa, aligned_f, METRICS_FILE)
	
	aggregate_and_save_final_results(METRICS_FILE, AGGREGATED_METRICS_FILE)

	print("=== CALIBRATION SYSTEM COMPLETE ===")

if __name__ == "__main__":
	main()