import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import numpy as np
import sys

try:
	from kneed import KneeLocator
	KNEED_AVAILABLE = True
except ImportError:
	KNEED_AVAILABLE = False

try:
	from scipy.stats import skew
except ImportError:
	print("Warning: 'scipy' not installed. 'skewness' feature will be 0.")
	print("Please run: pip install scipy")
	def skew(x):
		try:
			return 0.0 if hasattr(x, "__len__") else 0.0
		except Exception:
			return 0.0

COLS_TO_KEEP = {
	'fault_free_fa': ['timestamp', 'cal_pm25_fa'],
	'faulty_fa':	 ['timestamp', 'cal_pm25_fa'],
	'faulty_raw':	['timestamp', 'cal_pm25_0', 'cal_pm25_1',
					  'cal_pm25_2', 'cal_pm25_3']
}

VARIATIONS = ['fault_free_fa', 'faulty_fa', 'faulty_raw']
DayDict = Dict[int, pd.DataFrame]
DataStructure = Dict[str, Dict[str, DayDict]]
FeatureStructure = Dict[str, Dict[str, pd.DataFrame]]
LabelStructure = Dict[str, Dict[str, Dict[str, List[int]]]]
LabeledBin = Dict[str, DayDict]
FinalLabeledStructure = Dict[str, Dict[str, LabeledBin]]


def load_data(base_dir: str) -> (DataStructure, DataStructure, DataStructure):
	base_path = Path(base_dir)
	if not base_path.is_dir():
		print(f"Error: Base directory not found: {base_dir}")
		return {}, {}, {}

	fault_free_data: DataStructure = {}
	faulty_fa_data: DataStructure = {}
	faulty_raw_data: DataStructure = {}

	data_structs = {
		'fault_free_fa': fault_free_data,
		'faulty_fa': faulty_fa_data,
		'faulty_raw': faulty_raw_data
	}

	granularity_folders = [d for d in base_path.iterdir() if d.is_dir() and d.name in ['2.5', '5', '10', '20', '25', '50']]

	for gran_path in granularity_folders:
		gran_name = gran_path.name

		for struct in data_structs.values():
			struct[gran_name] = {}

		tf_folders = [tf for tf in gran_path.glob('TF_*') if tf.is_dir()]

		for tf_path in tf_folders:
			tf_name = tf_path.name

			for var_name in VARIATIONS:

				daily_data_dict: DayDict = {}

				for day in range(1, 31):
					file_name = f"{day}_{var_name}.csv"
					file_path = tf_path / file_name

					if file_path.exists():
						try:
							cols = COLS_TO_KEEP[var_name]
							df = pd.read_csv(file_path, usecols=cols)

							daily_data_dict[day] = df

						except Exception as e:
							print(f"Warning: Could not read {file_path}. Error: {e}")
					else:
						print(f"Info: File not found, skipping: {file_path}")
						pass

				if daily_data_dict:
					data_structs[var_name][gran_name][tf_name] = daily_data_dict
				else:
					print(f"Warning: No data found for {gran_name}/{tf_name}/{var_name}")

	print("Data loading complete.")
	return fault_free_data, faulty_fa_data, faulty_raw_data

def feature_extract(data_dict: DataStructure, data_type: str = 'fault_free') -> FeatureStructure:
	print(f"Starting feature extraction for: {data_type} (V4 - Clustering Optimized)...")
	feature_data: FeatureStructure = {}

	for gran_name, tf_dict in data_dict.items():
		feature_data[gran_name] = {}
		for tf_name, day_dict in tf_dict.items():

			if data_type == 'faulty_raw':
				sensor_dfs = {f'cal_pm25_{i}': [] for i in range(4)}

				for day_num, day_df in day_dict.items():
					raw_cols = [col for col in sensor_dfs.keys() if col in day_df.columns]
					if not raw_cols:
						print(f"Warning: Skipping {gran_name}/{tf_name}/Day {day_num} (Missing ALL raw cols)")
						continue

					for sensor_col in raw_cols:
						data_series = day_df[sensor_col].dropna()
						if len(data_series) < 5:
							print(f"Warning: Skipping {gran_name}/{tf_name}/Day {day_num}/{sensor_col} (Not enough valid data)")
							continue
						data_array = data_series.values.astype(float)
						N = len(data_array)

						mean_val = np.mean(data_array)
						std_val = np.std(data_array) if np.std(data_array) > 0 else 1e-6
						skew_val = skew(data_array)
						max_val = np.max(data_array)
						rms_val = np.sqrt(np.mean(data_array ** 2))
						crest_factor_val = max_val / rms_val if rms_val > 0 else 0.0
						high_thresh = mean_val + 2 * std_val
						high_val_pct = np.sum(data_array > high_thresh) / N
						slope_avg_val = np.mean(np.abs(np.diff(data_array))) if N > 1 else 0.0
						x = np.arange(N)
						try:
							coeffs = np.polyfit(x, data_array, 1)
							overall_slope_val = float(coeffs[0])
						except Exception:
							overall_slope_val = 0.0
						data_detrended = data_array - mean_val
						zero_cross_rate = np.sum(np.diff(np.sign(data_detrended)) != 0) / (N - 1) if N > 1 else 0.0
						fft_vals = np.fft.fft(data_detrended)
						fft_magnitudes = np.abs(fft_vals[1:(N // 2)]) if N // 2 > 1 else np.array([])
						max_ac_power_ratio = float(np.max(fft_magnitudes) / np.sum(fft_magnitudes)) if fft_magnitudes.size and np.sum(fft_magnitudes) > 0 else 0.0

						features = {
							'granularity': gran_name,
							'tf': tf_name,
							'day': int(day_num),
							'mean': float(mean_val),
							'std_dev': float(std_val),
							'skewness': float(skew_val),
							'crest_factor': float(crest_factor_val),
							'high_val_pct': float(high_val_pct),
							'slope_avg_local': float(slope_avg_val),
							'overall_slope_global': float(overall_slope_val),
							'zero_cross_rate': float(zero_cross_rate),
							'max_ac_power_ratio': float(max_ac_power_ratio),
						}
						sensor_dfs[sensor_col].append(features)

				feature_data[gran_name][tf_name] = {}
				for sensor_col, feats in sensor_dfs.items():
					if feats:
						df = pd.DataFrame(feats).set_index('day', drop=False)
					else:
						df = pd.DataFrame()
					feature_data[gran_name][tf_name][sensor_col] = df

			else:
				daily_features_list = []
				for day_num, day_df in day_dict.items():
					if 'cal_pm25_fa' not in day_df.columns:
						print(f"Warning: Skipping {gran_name}/{tf_name}/Day {day_num} (Missing cal_pm25_fa)")
						continue
					data_series = day_df['cal_pm25_fa'].dropna()
					if len(data_series) < 5:
						print(f"Warning: Skipping {gran_name}/{tf_name}/Day {day_num} (Not enough valid data)")
						continue
					data_array = data_series.values.astype(float)
					N = len(data_array)

					mean_val = np.mean(data_array)
					std_val = np.std(data_array) if np.std(data_array) > 0 else 1e-6
					skew_val = skew(data_array)
					max_val = np.max(data_array)
					rms_val = np.sqrt(np.mean(data_array ** 2))
					crest_factor_val = max_val / rms_val if rms_val > 0 else 0.0
					high_thresh = mean_val + 2 * std_val
					high_val_pct = np.sum(data_array > high_thresh) / N
					slope_avg_val = np.mean(np.abs(np.diff(data_array))) if N > 1 else 0.0
					x = np.arange(N)
					try:
						coeffs = np.polyfit(x, data_array, 1)
						overall_slope_val = float(coeffs[0])
					except Exception:
						overall_slope_val = 0.0
					data_detrended = data_array - mean_val
					zero_cross_rate = np.sum(np.diff(np.sign(data_detrended)) != 0) / (N - 1) if N > 1 else 0.0
					fft_vals = np.fft.fft(data_detrended)
					fft_magnitudes = np.abs(fft_vals[1:(N // 2)]) if N // 2 > 1 else np.array([])
					max_ac_power_ratio = float(np.max(fft_magnitudes) / np.sum(fft_magnitudes)) if fft_magnitudes.size and np.sum(fft_magnitudes) > 0 else 0.0

					features = {
						'granularity': gran_name,
						'tf': tf_name,
						'day': int(day_num),
						'mean': float(mean_val),
						'std_dev': float(std_val),
						'skewness': float(skew_val),
						'crest_factor': float(crest_factor_val),
						'high_val_pct': float(high_val_pct),
						'slope_avg_local': float(slope_avg_val),
						'overall_slope_global': float(overall_slope_val),
						'zero_cross_rate': float(zero_cross_rate),
						'max_ac_power_ratio': float(max_ac_power_ratio),
					}
					daily_features_list.append(features)

				if daily_features_list:
					feature_data[gran_name][tf_name] = pd.DataFrame(daily_features_list).set_index('day', drop=False)
				else:
					feature_data[gran_name][tf_name] = pd.DataFrame()
					print(f"Warning: No feature data generated for {gran_name}/{tf_name} ({data_type})")

	print(f"Feature extraction for {data_type} complete.")
	return feature_data


def label_days_automatic(
    feature_data: FeatureStructure,
    use_pca: bool = True,
    variance_threshold: float = 0.9,
    min_samples_fraction: float = 0.1,
    noise_factor: float = 1.0
) -> LabelStructure:

    FALLBACK_EPS = 1.5
    print(f"Starting *automatic* day labeling (PCA={use_pca}, threshold={variance_threshold})...")

    if not KNEED_AVAILABLE:
        print(f"Warning: 'kneed' library not found. Using fallback eps={FALLBACK_EPS}.")

    labels_data: LabelStructure = {}
    scaler = StandardScaler()

    for gran_name, tf_dict in feature_data.items():
        labels_data[gran_name] = {}

        for tf_name, data_df in tf_dict.items():
            if data_df is None or data_df.empty:
                labels_data[gran_name][tf_name] = {'N': [], 'A': []}
                continue

            numeric_df = data_df.select_dtypes(include=[np.number]).copy()
            if numeric_df.empty:
                all_days = data_df.index.tolist()
                labels_data[gran_name][tf_name] = {'N': [], 'A': all_days}
                continue

            invalid_mask = numeric_df.isnull().any(axis=1) | np.isinf(numeric_df.values).any(axis=1)
            days_to_drop = numeric_df[invalid_mask].index.tolist()
            numeric_clean = numeric_df[~invalid_mask]

            if numeric_clean.empty:
                all_days = data_df.index.tolist()
                labels_data[gran_name][tf_name] = {'N': [], 'A': all_days}
                continue

            INTERNAL_MIN_SAMPLES = max(2, int(min_samples_fraction * len(numeric_clean)))
            day_indices_clean = numeric_clean.index.tolist()

            try:
                data_scaled = scaler.fit_transform(numeric_clean)
            except Exception:
                all_days = data_df.index.tolist()
                labels_data[gran_name][tf_name] = {'N': [], 'A': all_days}
                continue

            if use_pca:
                pca_model = PCA(n_components=variance_threshold, svd_solver='full')
                data_reduced = pca_model.fit_transform(data_scaled)
            else:
                data_reduced = data_scaled

            auto_eps = FALLBACK_EPS
            if KNEED_AVAILABLE:
                try:
                    nn = NearestNeighbors(n_neighbors=INTERNAL_MIN_SAMPLES)
                    nn.fit(data_reduced)
                    distances, _ = nn.kneighbors(data_reduced)
                    k_distances = np.sort(distances[:, -1])

                    kneedle = KneeLocator(
                        x=list(range(1, len(k_distances) + 1)),
                        y=k_distances.tolist(),
                        curve='convex',
                        direction='increasing'
                    )

                    if getattr(kneedle, "elbow_y", None) is not None:
                        auto_eps = float(kneedle.elbow_y)
                except Exception:
                    auto_eps = FALLBACK_EPS

            auto_eps *= noise_factor

            db = DBSCAN(eps=auto_eps, min_samples=INTERNAL_MIN_SAMPLES)
            clusters = db.fit_predict(data_reduced)

            anomalous_days = list(days_to_drop)
            normal_days = []

            for i, label in enumerate(clusters):
                day = day_indices_clean[i]
                if label == -1:
                    anomalous_days.append(day)
                else:
                    normal_days.append(day)

            labels_data[gran_name][tf_name] = {
                'N': sorted(normal_days),
                'A': sorted(anomalous_days)
            }

    print("Automatic day labeling complete.")
    return labels_data

def label_days_manual(feature_data: FeatureStructure, eps: float = 1.5, min_samples: int = 3, use_pca: bool = True, variance_threshold: float = 0.95) -> LabelStructure:
	print(f"Starting *manual* day labeling (eps={eps}, min_samples={min_samples}, PCA={use_pca}, threshold={variance_threshold})...")

	labels_data: LabelStructure = {}
	scaler = StandardScaler()

	for gran_name, tf_dict in feature_data.items():
		labels_data[gran_name] = {}

		for tf_name, data_df in tf_dict.items():
			if data_df is None or data_df.empty:
				labels_data[gran_name][tf_name] = {'N': [], 'A': []}
				continue

			numeric_df = data_df.select_dtypes(include=[np.number]).copy()
			if numeric_df.empty:
				all_days = data_df.index.tolist()
				labels_data[gran_name][tf_name] = {'N': [], 'A': all_days}
				continue

			invalid_mask = numeric_df.isnull().any(axis=1) | np.isinf(numeric_df.values).any(axis=1)
			days_to_drop = numeric_df[invalid_mask].index.tolist()
			numeric_clean = numeric_df[~invalid_mask]

			if numeric_clean.empty or len(numeric_clean) < min_samples:
				all_days = data_df.index.tolist()
				labels_data[gran_name][tf_name] = {'N': [], 'A': all_days}
				continue

			day_indices_clean = numeric_clean.index.tolist()

			try:
				data_scaled = scaler.fit_transform(numeric_clean)
			except Exception:
				all_days = data_df.index.tolist()
				labels_data[gran_name][tf_name] = {'N': [], 'A': all_days}
				continue

			if use_pca:
				pca_model = PCA(n_components=variance_threshold, svd_solver='full')
				data_reduced = pca_model.fit_transform(data_scaled)
				n_comp = data_reduced.shape[1]
				explained = np.sum(pca_model.explained_variance_ratio_) * 100
				print(f"{gran_name}/{tf_name}: PCA reduced {data_scaled.shape[1]} → {n_comp} dims "
					  f"({explained:.1f}% variance explained)")
			else:
				data_reduced = data_scaled

			db = DBSCAN(eps=eps, min_samples=min_samples)
			clusters = db.fit_predict(data_reduced)

			anomalous_days = list(days_to_drop)
			normal_days = []

			for i, label in enumerate(clusters):
				day = day_indices_clean[i]
				if label == -1:
					anomalous_days.append(day)
				else:
					normal_days.append(day)

			labels_data[gran_name][tf_name] = {
				'N': sorted(normal_days),
				'A': sorted(anomalous_days)
			}

	print("Manual day labeling complete.")
	return labels_data

def _reorganize_one_set(source_day_dict: DayDict, normal_days: List[int], anomalous_days: List[int]) -> LabeledBin:

	new_data_bin: LabeledBin = {
		'normal': {},
		'anomalous': {}
	}

	for day in normal_days:
		if day in source_day_dict:
			new_data_bin['normal'][day] = source_day_dict[day]
		else:
			print(f"Info: Day {day} (Normal) not found in this data source. Skipping.")
			
			pass

	for day in anomalous_days:
		if day in source_day_dict:
			new_data_bin['anomalous'][day] = source_day_dict[day]
		else:
			print(f"Info: Day {day} (Anomalous) not found in this data source. Skipping.")
			pass

	return new_data_bin

def create_labeled_datasets(fault_free_data: DataStructure, faulty_data: DataStructure, faulty_raw_data: DataStructure, labels: LabelStructure) -> (FinalLabeledStructure, FinalLabeledStructure, FinalLabeledStructure):

	print("Starting to create labeled datasets...")
	

	labeled_free_data: FinalLabeledStructure = {}
	labeled_fa_data: FinalLabeledStructure = {}
	labeled_raw_data: FinalLabeledStructure = {}

	for gran_name, tf_dict in labels.items():

		labeled_free_data[gran_name] = {}
		labeled_fa_data[gran_name] = {}
		labeled_raw_data[gran_name] = {}

		for tf_name, label_info in tf_dict.items():

			normal_days = label_info.get('N', [])
			anomalous_days = label_info.get('A', [])

			try:
				source_dict = fault_free_data.get(gran_name, {}).get(tf_name, {})
				labeled_free_data[gran_name][tf_name] = _reorganize_one_set(
					source_dict, normal_days, anomalous_days
				)
			except KeyError:
				print(f"Warning: No source data in 'fault_free_data' for {gran_name}/{tf_name}")
				labeled_free_data[gran_name][tf_name] = {'normal': {}, 'anomalous': {}}

			try:
				source_dict = faulty_data.get(gran_name, {}).get(tf_name, {})
				labeled_fa_data[gran_name][tf_name] = _reorganize_one_set(
					source_dict, normal_days, anomalous_days
				)
			except KeyError:
				print(f"Warning: No source data in 'faulty_data' for {gran_name}/{tf_name}")
				labeled_fa_data[gran_name][tf_name] = {'normal': {}, 'anomalous': {}}

			try:
				source_dict = faulty_raw_data.get(gran_name, {}).get(tf_name, {})
				labeled_raw_data[gran_name][tf_name] = _reorganize_one_set(
					source_dict, normal_days, anomalous_days
				)

			except KeyError:
				print(f"Warning: No source data in 'faulty_raw_data' for {gran_name}/{tf_name}")
				labeled_raw_data[gran_name][tf_name] = {
					sensor: {'normal': {}, 'anomalous': {}} for sensor in [f'cal_pm25_{i}' for i in range(4)]
				}
				

	print("Labeled dataset creation complete.")
	return labeled_free_data, labeled_fa_data, labeled_raw_data

def save_labeled_datasets(output_dir: str,
						  labeled_free: FinalLabeledStructure,
						  labeled_fa: FinalLabeledStructure,
						  labeled_raw: FinalLabeledStructure):
	print("Saving labeled time-series datasets...")

	data_map = {
		'fault_free': labeled_free,
		'faulty_fa': labeled_fa,
		'faulty_raw': labeled_raw
	}

	base_path = Path(output_dir)

	for data_type, data_dict in data_map.items():
		for gran_name, tf_dict in data_dict.items():
			for tf_name, label_bin in tf_dict.items():
				for label_type, day_dict in label_bin.items():
					out_path = base_path / str(data_type) / str(gran_name) / str(tf_name) / str(label_type)
					out_path.mkdir(parents=True, exist_ok=True)
					for day_num, day_df in day_dict.items():
						file_path = out_path / f"day_{day_num}.csv"
						day_df.to_csv(file_path, index=False)
			   

	print("Time-series datasets saved successfully.")

def _save_all_labeled_features(output_dir: str, labels: LabelStructure, features_free: FeatureStructure, features_fa: FeatureStructure, features_raw: FeatureStructure):
	print("Saving all labeled feature data for visualization...")

	out_base_path = Path(output_dir) / "features_for_visualization"
	drop_cols = ["day", "granularity", "tf"]

	feature_map = {
		'fault_free': features_free,
		'faulty_fa': features_fa,
		'faulty_raw': features_raw
	}

	for gran_name, tf_dict in labels.items():
		for tf_name, label_info in tf_dict.items():
			out_path = out_base_path / gran_name
			out_path.mkdir(parents=True, exist_ok=True)

			normal_days = label_info.get('N', [])
			for data_type, feature_set in feature_map.items():
				if data_type == 'faulty_raw':
					tf_data = feature_set.get(gran_name, {}).get(tf_name, {})
					for sensor_name, feature_df in tf_data.items():
						if feature_df is None or feature_df.empty:
							print(f"Warning: No features found for {data_type}/{gran_name}/{tf_name}/{sensor_name}. Skipping.")
							continue

						df_to_save = feature_df.copy()
						df_to_save['label'] = 'Anomalous'
						df_to_save.loc[df_to_save.index.isin(normal_days), 'label'] = 'Normal'

						sensor_out_path = out_path / sensor_name
						sensor_out_path.mkdir(parents=True, exist_ok=True)

						file_name = f"{tf_name}_features_{data_type}_{sensor_name}.csv"
						file_path = sensor_out_path / file_name
						df_to_save.drop(columns=drop_cols, errors="ignore").to_csv(file_path, index=True)
				else:
					try:
						feature_df = feature_set.get(gran_name, {}).get(tf_name, pd.DataFrame())
					except Exception:
						feature_df = pd.DataFrame()

					if feature_df is None or feature_df.empty:
						print(f"Warning: No features found for {data_type}/{gran_name}/{tf_name}. Skipping save.")
						continue

					df_to_save = feature_df.copy()
					df_to_save['label'] = 'Anomalous'
					df_to_save.loc[df_to_save.index.isin(normal_days), 'label'] = 'Normal'

					file_name = f"{tf_name}_features_{data_type}.csv"
					file_path = out_path / file_name
					
					df_to_save.drop(columns=drop_cols, errors="ignore").to_csv(file_path, index=True)

	print("Labeled feature data saved.")

def save_all_outputs(output_dir: str, labeled_free: FinalLabeledStructure, labeled_fa: FinalLabeledStructure, labeled_raw: FinalLabeledStructure, features_free: FeatureStructure, features_fa: FeatureStructure, features_raw: FeatureStructure, labels: LabelStructure):

	print(f"Saving all project outputs to: {output_dir}")

	save_labeled_datasets(output_dir, labeled_free, labeled_fa, labeled_raw)

	_save_all_labeled_features(
		output_dir, labels,
		features_free, features_fa, features_raw
	)

	print("\n--- All saving operations complete. ---")

try:
	clustering_mode = sys.argv[1]
	if clustering_mode == "auto":
		variance_threshold = float(sys.argv[2])
		min_samples_fraction = float(sys.argv[3])
		noise_factor = float(sys.argv[4])
	elif clustering_mode == "manual":
		variance_threshold = float(sys.argv[2])
		eps = float(sys.argv[3])
		min_samples = int(sys.argv[4])
except:
	print("Enter the right number of input arguments.")
	sys.exit()
		

fault_free_data, faulty_fa_data, faulty_raw_data = load_data("Input")

fault_free_feature_data = feature_extract(fault_free_data, data_type='fault_free')
faulty_fa_feature_data = feature_extract(faulty_fa_data, data_type='faulty_fa')
faulty_raw_feature_data = feature_extract(faulty_raw_data, data_type='faulty_raw')


final_labels = label_days_automatic(fault_free_feature_data, use_pca = True, variance_threshold = variance_threshold, min_samples_fraction = min_samples_fraction, noise_factor = noise_factor)
#final_labels = label_days_manual(fault_free_feature_data, eps = eps, min_samples = min_samples, use_pca = True, variance_threshold = variance_threshold)

labeled_free_data, labeled_fa_data, labeled_raw_data = create_labeled_datasets(fault_free_data, faulty_fa_data, faulty_raw_data, final_labels)

save_all_outputs("Output", labeled_free_data, labeled_fa_data, labeled_raw_data, fault_free_feature_data, faulty_fa_feature_data, faulty_raw_feature_data, final_labels)
