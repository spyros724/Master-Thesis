from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np

# Min-Max scaling // per window
def train_data():
	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_Actuals.csv'
	df_all = pd.read_csv(path, index_col=0)

	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_TSOForecasts.csv'
	df_all_tso = pd.read_csv(path, index_col=0)

	countries_list = list(df_all.columns)
	print(countries_list)

	train_set_start = '2016-01-01'
	test_set_start = '2022-01-01'
	test_set_end = '2023-01-01'

	output_size = 36
	input_size = 168

	hour_diff = 12

	df_y_test = []
	df_x_test = []
	df_tso = []
	df_train = []
	df_train_details = []
	for country in tqdm(countries_list):
		# country = 'Estonia' # Austria Estonia Cyprus
		# Get country's data
		df_temp = df_all[country]
		df_temp_tso = df_all_tso[country]

		# Extract period of interest
		if country == 'Cyprus':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-06-09']
			
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-06-09']
		elif country == 'Ireland':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-03-07']

			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-03-07']
		else:
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < test_set_end]

			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < test_set_end]

		# Remove leading and trailing NANs
		first_idx = df_temp.first_valid_index()
		last_idx = df_temp.last_valid_index()
		df_temp = df_temp.loc[first_idx:last_idx]

		first_idx_tso = df_temp_tso.first_valid_index()
		last_idx_tso = df_temp_tso.last_valid_index()
		df_temp_tso = df_temp_tso.loc[first_idx_tso:last_idx_tso]

		# Generate train set
		# |--- > Calculate last index position for the train set
		#        |--- > (last train date - leading 12 hours of the forecast)
		last_train_idx = df_temp.loc[df_temp.index < test_set_start].index[-1]
		# idx_pos = df_temp.index.get_loc(last_train_idx) - (output_size - 24) + 1
		idx_pos = df_temp.index.get_loc(last_train_idx) - 24 + 1  # Clip the entire last day
		# |--- > Extract train series
		train_series = df_temp.iloc[:idx_pos]

		# |--- > Loop through the train part and create train set
		i = train_series.shape[0]
		while i >= input_size + output_size:
			# Extract series
			temp_series = train_series[(i - input_size - output_size):i]
			current_date = temp_series.index[-1]
			# Count nan values
			nan_count = temp_series.isna().sum()
			# If no nan values exist, then proceed
			if nan_count == 0:
				series_upper_limit = temp_series.mean() + 5 * temp_series.std()
				series_lower_limit = max(temp_series.mean() - 5 * temp_series.std(), 0)
				temp_series = np.clip(temp_series, a_min=series_lower_limit, a_max=series_upper_limit)

				# y_train
				y_train = np.asarray(temp_series[-output_size:])
				y_train = y_train.astype(float)

				# x_train
				x_train = np.asarray(temp_series[:input_size])
				x_train = x_train.astype(float)

				# Scale data
				scaler = MinMaxScaler(feature_range=(0, 1))
				x_train = scaler.fit_transform(np.reshape(x_train, (-1, 1))).reshape(-1)
				y_train = scaler.transform(np.reshape(y_train, (-1, 1))).reshape(-1)

				# Save sample
				out_train_details_line = [country, str(current_date), scaler.data_min_[0], scaler.data_max_[0]]
				df_train_details.append(out_train_details_line)
				out_train_line = list(x_train)
				out_train_line.extend(list(y_train))
				# out_train_line = np.asarray(out_train_line).reshape((1, (input_size + output_size + 5)))
				df_train.append(out_train_line)

			# Move to the previous day
			i = i - hour_diff

	df_train = np.asarray(df_train)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_' + str(hour_diff) + 'h_in' + str(input_size) + '.npy'
	np.save(export_path, df_train)

	df_train_details = np.asarray(df_train_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_details_' + str(hour_diff) + 'h_in' + str(input_size) + '.npy'
	np.save(export_path, df_train_details)

	print('Input size:', input_size)
	print('train file shape:', df_train.shape)
	print('train details file shape:', df_train_details.shape)
	print('-------------\n')

def test_data():
	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_Actuals.csv'
	df_all = pd.read_csv(path, index_col=0)

	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_TSOForecasts.csv'
	df_all_tso = pd.read_csv(path, index_col=0)

	countries_list = list(df_all.columns)
	print(countries_list)

	train_set_start = '2016-01-01'
	test_set_start = '2022-01-01'
	test_set_end = '2023-01-01'

	output_size = 36
	input_size = 168

	hour_diff = 12

	df_y_test = []
	df_x_test = []
	df_tso = []
	df_train = []
	df_train_details = []
	for country in tqdm(countries_list):
		# country = 'Estonia' # Austria Estonia Cyprus
		# Get country's data
		df_temp = df_all[country]
		df_temp_tso = df_all_tso[country]

		# Extract period of interest
		if country == 'Cyprus':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-06-09']
			
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-06-09']
		elif country == 'Ireland':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-03-07']

			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-03-07']
		else:
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < test_set_end]

			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < test_set_end]

		# Remove leading and trailing NANs
		first_idx = df_temp.first_valid_index()
		last_idx = df_temp.last_valid_index()
		df_temp = df_temp.loc[first_idx:last_idx]

		first_idx_tso = df_temp_tso.first_valid_index()
		last_idx_tso = df_temp_tso.last_valid_index()
		df_temp_tso = df_temp_tso.loc[first_idx_tso:last_idx_tso]

		# Generate test set
		# |--- > Calculate first index position for the test set
		#        |--- > (First test date - leading 12 hours of the forecast - number of past observations)
		first_test_idx = df_temp.loc[df_temp.index > test_set_start].index[0]
		idx_pos = df_temp.index.get_loc(first_test_idx) - (output_size - 24) - input_size

		first_test_idx_tso = df_temp_tso.loc[df_temp_tso.index > test_set_start].index[0]
		idx_pos_tso = df_temp_tso.index.get_loc(first_test_idx_tso) - (output_size - 24) - input_size
		# |--- > Extract test series
		test_series = df_temp.iloc[idx_pos:]
		test_series_tso = df_temp_tso.iloc[idx_pos_tso:]

		# Fill NAs for the test set
		max_cons_nans = test_series.isnull().astype(int).groupby(test_series.notnull().astype(int).cumsum()).sum().max()
		nan_count = test_series.isna().sum()
		if nan_count > 0:
			for i in range(test_series.shape[0]):
				if np.isnan(test_series.iloc[i]):
					# Check if out of bounds
					repl_idx_min = i - 168
					repl_idx_max = i + 168
					if (repl_idx_min >= 0) & (repl_idx_max < test_series.shape[0]):
						# Replace with same values based on day of week
						test_series.iloc[i] = (test_series.iloc[i - 168] + test_series.iloc[i + 168]) / 2
						if np.isnan(test_series.iloc[i]):
							test_series.iloc[i] = (test_series.iloc[i - 24] + test_series.iloc[i + 24]) / 2
					else:
						# Replace with same values based on previous day
						repl_idx_min = i - 24
						repl_idx_max = i + 24
						if (repl_idx_min >= 0) & (repl_idx_max < test_series.shape[0]):
							test_series.iloc[i] = (test_series.iloc[i - 24] + test_series.iloc[i + 24]) / 2
						else:
							# Replace with same values based on previous / next hour
							test_series.iloc[i] = (test_series.iloc[i - 1] + test_series.iloc[i + 1]) / 2
			# Second pass for filling nans
			for i in range(test_series.shape[0]):
				if np.isnan(test_series.iloc[i]):
					test_series.iloc[i] = (test_series.iloc[i - 1] + test_series.iloc[i + 1]) / 2
		
		# |--- > Loop through the test part and create x_test and y_test sets
		# Find last test idx - Initialize the loop
		max_date = max(test_series.index[test_series.index.str.contains('23:00:00')])
		i = test_series.index.get_loc(max_date)
		# Implement the loop
		while i > input_size + output_size - 2:
			# Extract series
			temp_series = test_series[(i - input_size - output_size + 1):(i + 1)]
			current_date = temp_series.index[-1]

			# y_test
			y_test = np.asarray(temp_series[-output_size:])
			y_test = y_test.astype(float)
			# Save samples
			out_y_test_line = [(i + 1), country, str(current_date)]
			out_y_test_line.extend(list(y_test))
			df_y_test.append(out_y_test_line)
		
			# x_test
			x_test = np.asarray(temp_series[:input_size])
			x_test = x_test.astype(float)
			# Scale data
			scaler = MinMaxScaler(feature_range=(0, 1))
			

			out_x_test_line = [(i + 1), country, str(current_date), scaler.data_min_[0], scaler.data_max_[0]]
			out_x_test_line.extend(list(x_test))
			df_x_test.append(out_x_test_line)
		
			# Move to the previous day
			i = i - 24

		# |--- > Loop through the test part and create TSOs' forecasts
		# Find last test idx - Initialize the loop
		max_date = max(test_series_tso.index[test_series_tso.index.str.contains('23:00:00')])
		i = test_series_tso.index.get_loc(max_date)
		# Implement the loop
		while i > input_size + output_size - 2:
			# Extract series
			temp_series = test_series_tso[(i - input_size - output_size + 1):(i + 1)]
			current_date = temp_series.index[-1]

			# Extract seriesprint('x test file shape:', df_x_test.shape)
			# print('y test file shape:', df_y_test.shape)
			# y_test
			y_test = np.asarray(temp_series[-output_size:])
			y_test = y_test.astype(float)

			df_tso.append(out_y_test_line)
		
			# Move to the previous day
			i = i - 24

	df_tso = np.asarray(df_tso)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/tso_forecasts.npy'
	np.save(export_path, df_tso)

	df_x_test = np.asarray(df_x_test)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/x_test_in' + str(input_size) + '.npy'
	np.save(export_path, df_x_test)

	df_y_test = np.asarray(df_y_test)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/y_test.npy'
	np.save(export_path, df_y_test)

	print('x test file shape:', df_x_test.shape)
	print('y test file shape:', df_y_test.shape)
	print('tso file shape:', df_tso.shape)
	print('-------------\n')


# No scaling
def train_data_nosc():
	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_Actuals.csv'
	df_all = pd.read_csv(path, index_col=0)

	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_TSOForecasts.csv'
	df_all_tso = pd.read_csv(path, index_col=0)

	countries_list = list(df_all.columns)
	print(countries_list)

	train_set_start = '2016-01-01'
	test_set_start = '2022-01-01'
	test_set_end = '2023-01-01'

	output_size = 36
	input_size = 168

	hour_diff = 12

	df_y_test = []
	df_x_test = []
	df_tso = []
	df_train = []
	df_train_details = []
	for country in tqdm(countries_list):
		# country = 'Estonia' # Austria Estonia Cyprus
		# Get country's data
		df_temp = df_all[country]
		df_temp_tso = df_all_tso[country]

		# Extract period of interest
		if country == 'Cyprus':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-06-09']
			
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-06-09']
		elif country == 'Ireland':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-03-07']

			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-03-07']
		else:
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < test_set_end]

			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < test_set_end]

		# Remove leading and trailing NANs
		first_idx = df_temp.first_valid_index()
		last_idx = df_temp.last_valid_index()
		df_temp = df_temp.loc[first_idx:last_idx]

		first_idx_tso = df_temp_tso.first_valid_index()
		last_idx_tso = df_temp_tso.last_valid_index()
		df_temp_tso = df_temp_tso.loc[first_idx_tso:last_idx_tso]

		# Generate train set
		# |--- > Calculate last index position for the train set
		#        |--- > (last train date - leading 12 hours of the forecast)
		last_train_idx = df_temp.loc[df_temp.index < test_set_start].index[-1]
		# idx_pos = df_temp.index.get_loc(last_train_idx) - (output_size - 24) + 1
		idx_pos = df_temp.index.get_loc(last_train_idx) - 24 + 1  # Clip the entire last day
		# |--- > Extract train series
		train_series = df_temp.iloc[:idx_pos]

		# |--- > Loop through the train part and create train set
		i = train_series.shape[0]
		while i >= input_size + output_size:
			# Extract series
			temp_series = train_series[(i - input_size - output_size):i]
			current_date = temp_series.index[-1]
			# Count nan values
			nan_count = temp_series.isna().sum()
			# If no nan values exist, then proceed
			if nan_count == 0:
				series_upper_limit = temp_series.mean() + 5 * temp_series.std()
				series_lower_limit = max(temp_series.mean() - 5 * temp_series.std(), 0)
				temp_series = np.clip(temp_series, a_min=series_lower_limit, a_max=series_upper_limit)

				# y_train
				y_train = np.asarray(temp_series[-output_size:])
				y_train = y_train.astype(float)

				# x_train
				x_train = np.asarray(temp_series[:input_size])
				x_train = x_train.astype(float)

				# # Scale data
				# scaler = MinMaxScaler(feature_range=(0, 1))
				# x_train = scaler.fit_transform(np.reshape(x_train, (-1, 1))).reshape(-1)
				# y_train = scaler.transform(np.reshape(y_train, (-1, 1))).reshape(-1)

				# Save sample
				out_train_details_line = [country, str(current_date), 0, 0]
				df_train_details.append(out_train_details_line)
				out_train_line = list(x_train)
				out_train_line.extend(list(y_train))
				# out_train_line = np.asarray(out_train_line).reshape((1, (input_size + output_size + 5)))
				df_train.append(out_train_line)

			# Move to the previous day
			i = i - hour_diff

	df_train = np.asarray(df_train)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_' + str(hour_diff) + 'h_in' + str(input_size) + '_nosc.npy'
	np.save(export_path, df_train)

	df_train_details = np.asarray(df_train_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_details_' + str(hour_diff) + 'h_in' + str(input_size) + '_nosc.npy'
	np.save(export_path, df_train_details)

	print('Input size:', input_size)
	print('train file shape:', df_train.shape)
	print('train details file shape:', df_train_details.shape)
	print('-------------\n')

def test_data_nosc():
	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_Actuals.csv'
	df_all = pd.read_csv(path, index_col=0)

	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_TSOForecasts.csv'
	df_all_tso = pd.read_csv(path, index_col=0)

	countries_list = list(df_all.columns)
	print(countries_list)

	train_set_start = '2016-01-01'
	test_set_start = '2022-01-01'
	test_set_end = '2023-01-01'

	output_size = 36
	input_size = 168

	hour_diff = 12

	df_y_test = []
	df_x_test = []
	df_tso = []
	df_train = []
	df_train_details = []
	for country in tqdm(countries_list):
		# country = 'Estonia' # Austria Estonia Cyprus
		# Get country's data
		df_temp = df_all[country]
		df_temp_tso = df_all_tso[country]

		# Extract period of interest
		if country == 'Cyprus':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-06-09']
			
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-06-09']
		elif country == 'Ireland':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-03-07']

			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-03-07']
		else:
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < test_set_end]

			df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			df_temp_tso = df_temp_tso.loc[df_temp_tso.index < test_set_end]

		# Remove leading and trailing NANs
		first_idx = df_temp.first_valid_index()
		last_idx = df_temp.last_valid_index()
		df_temp = df_temp.loc[first_idx:last_idx]

		first_idx_tso = df_temp_tso.first_valid_index()
		last_idx_tso = df_temp_tso.last_valid_index()
		df_temp_tso = df_temp_tso.loc[first_idx_tso:last_idx_tso]

		# Generate test set
		# |--- > Calculate first index position for the test set
		#        |--- > (First test date - leading 12 hours of the forecast - number of past observations)
		first_test_idx = df_temp.loc[df_temp.index > test_set_start].index[0]
		idx_pos = df_temp.index.get_loc(first_test_idx) - (output_size - 24) - input_size

		first_test_idx_tso = df_temp_tso.loc[df_temp_tso.index > test_set_start].index[0]
		idx_pos_tso = df_temp_tso.index.get_loc(first_test_idx_tso) - (output_size - 24) - input_size
		# |--- > Extract test series
		test_series = df_temp.iloc[idx_pos:]
		test_series_tso = df_temp_tso.iloc[idx_pos_tso:]

		# Fill NAs for the test set
		max_cons_nans = test_series.isnull().astype(int).groupby(test_series.notnull().astype(int).cumsum()).sum().max()
		nan_count = test_series.isna().sum()
		if nan_count > 0:
			for i in range(test_series.shape[0]):
				if np.isnan(test_series.iloc[i]):
					# Check if out of bounds
					repl_idx_min = i - 168
					repl_idx_max = i + 168
					if (repl_idx_min >= 0) & (repl_idx_max < test_series.shape[0]):
						# Replace with same values based on day of week
						test_series.iloc[i] = (test_series.iloc[i - 168] + test_series.iloc[i + 168]) / 2
						if np.isnan(test_series.iloc[i]):
							test_series.iloc[i] = (test_series.iloc[i - 24] + test_series.iloc[i + 24]) / 2
					else:
						# Replace with same values based on previous day
						repl_idx_min = i - 24
						repl_idx_max = i + 24
						if (repl_idx_min >= 0) & (repl_idx_max < test_series.shape[0]):
							test_series.iloc[i] = (test_series.iloc[i - 24] + test_series.iloc[i + 24]) / 2
						else:
							# Replace with same values based on previous / next hour
							test_series.iloc[i] = (test_series.iloc[i - 1] + test_series.iloc[i + 1]) / 2
			# Second pass for filling nans
			for i in range(test_series.shape[0]):
				if np.isnan(test_series.iloc[i]):
					test_series.iloc[i] = (test_series.iloc[i - 1] + test_series.iloc[i + 1]) / 2
		
		# |--- > Loop through the test part and create x_test and y_test sets
		# Find last test idx - Initialize the loop
		max_date = max(test_series.index[test_series.index.str.contains('23:00:00')])
		i = test_series.index.get_loc(max_date)
		# Implement the loop
		while i > input_size + output_size - 2:
			# Extract series
			temp_series = test_series[(i - input_size - output_size + 1):(i + 1)]
			current_date = temp_series.index[-1]

			# y_test
			y_test = np.asarray(temp_series[-output_size:])
			y_test = y_test.astype(float)
			# Save samples
			out_y_test_line = [(i + 1), country, str(current_date)]
			out_y_test_line.extend(list(y_test))
			df_y_test.append(out_y_test_line)
		
			# x_test
			x_test = np.asarray(temp_series[:input_size])
			x_test = x_test.astype(float)		

			out_x_test_line = [(i + 1), country, str(current_date), 0, 0]
			out_x_test_line.extend(list(x_test))
			df_x_test.append(out_x_test_line)
		
			# Move to the previous day
			i = i - 24

		# |--- > Loop through the test part and create TSOs' forecasts
		# Find last test idx - Initialize the loop
		max_date = max(test_series_tso.index[test_series_tso.index.str.contains('23:00:00')])
		i = test_series_tso.index.get_loc(max_date)
		# Implement the loop
		while i > input_size + output_size - 2:
			# Extract series
			temp_series = test_series_tso[(i - input_size - output_size + 1):(i + 1)]
			current_date = temp_series.index[-1]

			# Extract seriesprint('x test file shape:', df_x_test.shape)
			# print('y test file shape:', df_y_test.shape)
			# y_test
			y_test = np.asarray(temp_series[-output_size:])
			y_test = y_test.astype(float)

			df_tso.append(out_y_test_line)
		
			# Move to the previous day
			i = i - 24

	df_tso = np.asarray(df_tso)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/tso_forecasts.npy'
	np.save(export_path, df_tso)

	df_x_test = np.asarray(df_x_test)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/x_test_in' + str(input_size) + '_nosc.npy'
	np.save(export_path, df_x_test)

	df_y_test = np.asarray(df_y_test)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/y_test.npy'
	np.save(export_path, df_y_test)

	print('x test file shape:', df_x_test.shape)
	print('y test file shape:', df_y_test.shape)
	print('tso file shape:', df_tso.shape)
	print('-------------\n')


# Normalization // per country
def train_data_normalize_per_country():
	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_Actuals.csv'
	df_all = pd.read_csv(path, index_col=0)

	# path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_TSOForecasts.csv'
	# df_all_tso = pd.read_csv(path, index_col=0)

	countries_list = list(df_all.columns)
	print(countries_list)

	train_set_start = '2016-01-01'
	test_set_start = '2022-01-01'
	test_set_end = '2023-01-01'

	output_size = 36
	input_size = 168

	hour_diff = 24

	df_y_test = []
	df_x_test = []
	# df_tso = []
	df_train = []
	df_train_details = []
	for country in tqdm(countries_list):
		# country = 'Estonia' # Austria Estonia Cyprus
		# Get country's data
		df_temp = df_all[country]
		# df_temp_tso = df_all_tso[country]

		# Extract period of interest
		if country == 'Cyprus':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-06-09']
			
			# df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			# df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-06-09']
		elif country == 'Ireland':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-03-07']

			# df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			# df_temp_tso = df_temp_tso.loc[df_temp_tso.index < '2022-03-07']
		else:
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < test_set_end]

			# df_temp_tso = df_temp_tso.loc[df_temp_tso.index > train_set_start]
			# df_temp_tso = df_temp_tso.loc[df_temp_tso.index < test_set_end]

		# Remove leading and trailing NANs
		first_idx = df_temp.first_valid_index()
		last_idx = df_temp.last_valid_index()
		df_temp = df_temp.loc[first_idx:last_idx]

		# Generate train set
		# |--- > Calculate last index position for the train set
		#        |--- > (last train date - leading 12 hours of the forecast)
		last_train_idx = df_temp.loc[df_temp.index < test_set_start].index[-1]
		# idx_pos = df_temp.index.get_loc(last_train_idx) - (output_size - 24) + 1
		idx_pos = df_temp.index.get_loc(last_train_idx) - 24 + 1  # Clip the entire last day
		# |--- > Extract train series
		train_series = df_temp.iloc[:idx_pos]
		# |--- > Eliminate outliers
		ll = np.nanquantile(train_series, 0.001)
		ul = np.nanquantile(train_series, 0.999)
		train_series = train_series.clip(lower=ll, upper=ul)

		# |--- > Standardize the series
		scaler_mean = train_series.mean()
		scaler_std = train_series.std()
		train_series = (train_series - scaler_mean) / scaler_std

		# |--- > Loop through the train part and create train set
		i = train_series.shape[0]
		while i >= input_size + output_size:
			# Extract series
			temp_series = train_series[(i - input_size - output_size):i]
			current_date = temp_series.index[-1]
			# Count nan values
			nan_count = temp_series.isna().sum()
			# If no nan values exist, then proceed
			if nan_count == 0:
				# series_upper_limit = temp_series.mean() + 5 * temp_series.std()
				# series_lower_limit = max(temp_series.mean() - 5 * temp_series.std(), 0)
				# temp_series = np.clip(temp_series, a_min=series_lower_limit, a_max=series_upper_limit)

				# y_train
				y_train = np.asarray(temp_series[-output_size:])
				y_train = y_train.astype(float).reshape(-1)

				# x_train
				x_train = np.asarray(temp_series[:input_size])
				x_train = x_train.astype(float).reshape(-1)

				# # Scale data
				# scaler = MinMaxScaler(feature_range=(0, 1))
				# x_train = scaler.fit_transform(np.reshape(x_train, (-1, 1))).reshape(-1)
				# y_train = scaler.transform(np.reshape(y_train, (-1, 1))).reshape(-1)

				# Save sample
				out_train_details_line = [country, str(current_date), scaler_mean, scaler_std]
				df_train_details.append(out_train_details_line)
				out_train_line = list(x_train)
				out_train_line.extend(list(y_train))
				df_train.append(out_train_line)

			# Move to the previous day
			i = i - hour_diff

	df_train = np.asarray(df_train)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_' + str(hour_diff) + 'h_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, df_train)

	df_train_details = np.asarray(df_train_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_details_' + str(hour_diff) + 'h_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, df_train_details)

	print('Input size:', input_size)
	print('train file shape:', df_train.shape)
	print('train details file shape:', df_train_details.shape)
	print('-------------\n')

def test_data_normalize_per_country():
	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_AllData_Actuals.csv'
	df_all = pd.read_csv(path, index_col=0)

	countries_list = list(df_all.columns)
	print(countries_list)

	train_set_start = '2016-01-01'
	test_set_start = '2022-01-01'
	test_set_end = '2023-01-01'

	output_size = 36
	input_size = 168

	hour_diff = 24

	df_y_test = []
	df_x_test = []
	df_train = []
	df_train_details = []
	for country in tqdm(countries_list):
		# Get country's data
		df_temp = df_all[country]

		# Extract period of interest
		if country == 'Cyprus':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-06-09']
		elif country == 'Ireland':
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < '2022-03-07']
		else:
			df_temp = df_temp.loc[df_temp.index > train_set_start]
			df_temp = df_temp.loc[df_temp.index < test_set_end]

		# Remove leading and trailing NANs
		first_idx = df_temp.first_valid_index()
		last_idx = df_temp.last_valid_index()
		df_temp = df_temp.loc[first_idx:last_idx]

		# Generate train set
		# |--- > Calculate last index position for the train set
		#        |--- > (last train date - leading 12 hours of the forecast)
		last_train_idx = df_temp.loc[df_temp.index < test_set_start].index[-1]
		# idx_pos = df_temp.index.get_loc(last_train_idx) - (output_size - 24) + 1
		idx_pos = df_temp.index.get_loc(last_train_idx) - 24 + 1  # Clip the entire last day
		
		# |--- > Extract train series
		train_series = df_temp.iloc[:idx_pos]
		# |--- > Eliminate outliers
		ll = np.nanquantile(train_series, 0.001)
		ul = np.nanquantile(train_series, 0.999)
		train_series = train_series.clip(lower=ll, upper=ul)

		# |--- > Calculate values for standardizing the series
		scaler_mean = train_series.mean()
		scaler_std = train_series.std()

		# Generate test set
		# |--- > Calculate first index position for the test set
		#        |--- > (First test date - leading 12 hours of the forecast - number of past observations)
		first_test_idx = df_temp.loc[df_temp.index > test_set_start].index[0]
		idx_pos = df_temp.index.get_loc(first_test_idx) - (output_size - 24) - input_size

		# |--- > Extract test series
		test_series = df_temp.iloc[idx_pos:]
		test_series = (test_series - scaler_mean) / scaler_std
		# test_series_tso = df_temp_tso.iloc[idx_pos_tso:]

		# Fill NAs for the test set
		max_cons_nans = test_series.isnull().astype(int).groupby(test_series.notnull().astype(int).cumsum()).sum().max()
		nan_count = test_series.isna().sum()
		if nan_count > 0:
			for i in range(test_series.shape[0]):
				if np.isnan(test_series.iloc[i]):
					# Check if out of bounds
					repl_idx_min = i - 168
					repl_idx_max = i + 168
					if (repl_idx_min >= 0) & (repl_idx_max < test_series.shape[0]):
						# Replace with same values based on day of week
						test_series.iloc[i] = (test_series.iloc[i - 168] + test_series.iloc[i + 168]) / 2
						if np.isnan(test_series.iloc[i]):
							test_series.iloc[i] = (test_series.iloc[i - 24] + test_series.iloc[i + 24]) / 2
					else:
						# Replace with same values based on previous day
						repl_idx_min = i - 24
						repl_idx_max = i + 24
						if (repl_idx_min >= 0) & (repl_idx_max < test_series.shape[0]):
							test_series.iloc[i] = (test_series.iloc[i - 24] + test_series.iloc[i + 24]) / 2
						else:
							# Replace with same values based on previous / next hour
							test_series.iloc[i] = (test_series.iloc[i - 1] + test_series.iloc[i + 1]) / 2
			# Second pass for filling nans
			for i in range(test_series.shape[0]):
				if np.isnan(test_series.iloc[i]):
					test_series.iloc[i] = (test_series.iloc[i - 1] + test_series.iloc[i + 1]) / 2
		

		# |--- > Loop through the test part and create x_test and y_test sets
		# Find last test idx - Initialize the loop
		max_date = max(test_series.index[test_series.index.str.contains('23:00:00')])
		i = test_series.index.get_loc(max_date)
		# Implement the loop
		while i > input_size + output_size - 2:
			# Extract series
			temp_series = test_series[(i - input_size - output_size + 1):(i + 1)]
			current_date = temp_series.index[-1]

			# y_test
			y_test = np.asarray(temp_series[-output_size:])
			y_test = y_test.astype(float)
			# Save samples
			out_y_test_line = [(i + 1), country, str(current_date)]
			out_y_test_line.extend(list(y_test))
			df_y_test.append(out_y_test_line)
		
			# x_test
			x_test = np.asarray(temp_series[:input_size])
			x_test = x_test.astype(float)	
			out_x_test_line = [(i + 1), country, str(current_date), scaler_mean, scaler_std]
			out_x_test_line.extend(list(x_test))
			df_x_test.append(out_x_test_line)
		
			# Move to the previous day
			i = i - 24

	df_x_test = np.asarray(df_x_test)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/x_test_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, df_x_test)

	df_y_test = np.asarray(df_y_test)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/y_test_counorm.npy'
	np.save(export_path, df_y_test)

	print('x test file shape:', df_x_test.shape)
	print('y test file shape:', df_y_test.shape)
	print('-------------\n')


# Exogenous variables
def exogenous_calendar_cosine():
	# Params
	output_size = 36
	input_size = 168
	hour_diff = 12

	# Train
	path = 'Data/ENTSOE_Energy_Load/Processed/train_details_' + str(hour_diff) + 'h_in' + str(input_size) + '.npy'
	train_details = np.load(path)
	print(train_details.shape)
	print(train_details[0, :])

	# Ordinal
	train_details = pd.DataFrame(train_details[:, :2], columns=['Country', 'Datetime'])
	train_details['Datetime'] = pd.to_datetime(train_details['Datetime'])
	train_details['Hour'] = train_details['Datetime'].dt.hour
	train_details['Weekday'] = train_details['Datetime'].dt.dayofweek
	train_details['Month'] = train_details['Datetime'].dt.month
	train_details = train_details.drop(columns=['Datetime', 'Country'])

	# Cosine
	train_details['Hour_sin'] = np.sin(2*np.pi*train_details['Hour']/24)
	train_details['Hour_cos'] = np.cos(2*np.pi*train_details['Hour']/24)
	train_details['Weekday_sin'] = np.sin(2*np.pi*train_details['Weekday']/7)
	train_details['Weekday_cos'] = np.cos(2*np.pi*train_details['Weekday']/7)
	train_details['Month_sin'] = np.sin(2*np.pi*train_details['Month']/12)
	train_details['Month_cos'] = np.cos(2*np.pi*train_details['Month']/12)
	train_details = train_details.drop(columns=['Hour', 'Weekday', 'Month'])

	# train_details['Year'] = train_details['Datetime'].dt.year
	# train_details['Monthday'] = train_details['Datetime'].dt.day
	# train_details['Week'] = train_details['Datetime'].dt.isocalendar().week

	print(train_details.iloc[0, :])

	train_details = np.asarray(train_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_calendar_cosin_' + str(hour_diff) + 'h_in' + str(input_size) + '.npy'
	np.save(export_path, train_details)
	print(train_details.shape)


	# Test
	path = 'Data/ENTSOE_Energy_Load/Processed/x_test_in' + str(input_size) + '.npy'
	test_details = np.load(path)
	test_details = pd.DataFrame(test_details[:, 1:3], columns=['Country', 'Datetime'])
	print(test_details.iloc[0, :])
	print(test_details.shape)

	# Ordinal
	test_details['Datetime'] = pd.to_datetime(test_details['Datetime'])
	test_details['Hour'] = test_details['Datetime'].dt.hour
	test_details['Weekday'] = test_details['Datetime'].dt.dayofweek
	test_details['Month'] = test_details['Datetime'].dt.month
	test_details = test_details.drop(columns=['Datetime', 'Country'])

	# Cosine
	test_details['Hour_sin'] = np.sin(2*np.pi*test_details['Hour']/24)
	test_details['Hour_cos'] = np.cos(2*np.pi*test_details['Hour']/24)
	test_details['Weekday_sin'] = np.sin(2*np.pi*test_details['Weekday']/7)
	test_details['Weekday_cos'] = np.cos(2*np.pi*test_details['Weekday']/7)
	test_details['Month_sin'] = np.sin(2*np.pi*test_details['Month']/12)
	test_details['Month_cos'] = np.cos(2*np.pi*test_details['Month']/12)
	test_details = test_details.drop(columns=['Hour', 'Weekday', 'Month'])


	print(test_details.iloc[0, :])


	test_details = np.asarray(test_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/x_test_calendar_cosin_in' + str(input_size) + '.npy'
	np.save(export_path, test_details)
	print(test_details.shape)


def exogenous_calendar_onehot():
	# Params
	output_size = 36
	input_size = 168
	hour_diff = 24

	# Train
	path = 'Data/ENTSOE_Energy_Load/Processed/train_details_' + str(hour_diff) + 'h_in' + str(input_size) + '_counorm.npy'
	train_details = np.load(path)

	# Ordinal
	train_details = pd.DataFrame(train_details[:, :2], columns=['Country', 'Datetime'])
	train_details['Datetime'] = pd.to_datetime(train_details['Datetime'])
	train_details['Hour'] = train_details['Datetime'].dt.hour
	train_details['Weekday'] = train_details['Datetime'].dt.dayofweek
	train_details['Month'] = train_details['Datetime'].dt.month
	train_details = train_details.drop(columns=['Datetime', 'Country'])

	if hour_diff == 12:
		df_hour = pd.get_dummies(train_details['Hour'], prefix='h')
		df_hour = np.asarray(df_hour)
		df_weekday = pd.get_dummies(train_details['Weekday'], prefix='d')
		df_weekday = np.asarray(df_weekday)
		df_month = pd.get_dummies(train_details['Month'], prefix='m')
		df_month = np.asarray(df_month)
		df_out = np.hstack((df_hour, df_weekday, df_month))
	elif hour_diff == 24:
		df_weekday = pd.get_dummies(train_details['Weekday'], prefix='d')
		df_weekday = np.asarray(df_weekday)
		df_month = pd.get_dummies(train_details['Month'], prefix='m')
		df_month = np.asarray(df_month)
		df_out = np.hstack((df_weekday, df_month))

	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_calendar_onehot_' + str(hour_diff) + 'h_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, df_out)
	print(df_out.shape)


	# Test
	path = 'Data/ENTSOE_Energy_Load/Processed/x_test_in' + str(input_size) + '_counorm.npy'
	test_details = np.load(path)
	test_details = pd.DataFrame(test_details[:, 1:3], columns=['Country', 'Datetime'])

	# Ordinal
	test_details['Datetime'] = pd.to_datetime(test_details['Datetime'])
	test_details['Hour'] = test_details['Datetime'].dt.hour
	test_details['Weekday'] = test_details['Datetime'].dt.dayofweek
	test_details['Month'] = test_details['Datetime'].dt.month
	test_details = test_details.drop(columns=['Datetime', 'Country'])

	if hour_diff == 12:
		df_hour = pd.get_dummies(test_details['Hour'], prefix='h')
		df_hour['h_11'] = 0
		df_hour = df_hour[['h_11', 'h_23']]
		df_weekday = pd.get_dummies(test_details['Weekday'], prefix='d')
		df_weekday = np.asarray(df_weekday)
		df_month = pd.get_dummies(test_details['Month'], prefix='m')
		df_month = np.asarray(df_month)
		df_out = np.hstack((df_hour, df_weekday, df_month))
	elif hour_diff == 24:
		df_weekday = pd.get_dummies(test_details['Weekday'], prefix='d')
		df_weekday = np.asarray(df_weekday)
		df_month = pd.get_dummies(test_details['Month'], prefix='m')
		df_month = np.asarray(df_month)
		df_out = np.hstack((df_weekday, df_month))

	export_path = 'Data/ENTSOE_Energy_Load/Processed/x_test_calendar_onehot_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, df_out)
	print(df_out.shape)


def exogenous_countries_onehot():
	# Params
	output_size = 36
	input_size = 168
	hour_diff = 24

	# Train
	path = 'Data/ENTSOE_Energy_Load/Processed/train_details_' + str(hour_diff) + 'h_in' + str(input_size) + '_counorm.npy'
	train_details = np.load(path)

	# One hot country
	train_details = pd.DataFrame(train_details[:, :2], columns=['Country', 'Datetime'])
	df_out = pd.get_dummies(train_details['Country'])
	df_out = np.asarray(df_out)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_country_onehot_' + str(hour_diff) + 'h_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, df_out)
	print(df_out.shape)

	# Test
	path = 'Data/ENTSOE_Energy_Load/Processed/x_test_in' + str(input_size) + '_counorm.npy'
	test_details = np.load(path)
	test_details = pd.DataFrame(test_details[:, 1:3], columns=['Country', 'Datetime'])

	# One hot country
	df_out = pd.get_dummies(test_details['Country'])
	df_out = np.asarray(df_out)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/x_test_country_onehot_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, df_out)
	print(df_out.shape)
	

def exogenous_temperatures_raw():
	# Params
	output_size = 36
	input_size = 168
	hour_diff = 12

	# Weather data 
	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_WeatherData.csv'
	df_weather = pd.read_csv(path)
	df_weather['Datetime'] = pd.to_datetime(df_weather['Unnamed: 0'])
	df_weather['Date'] = df_weather['Datetime'].dt.date
	df_weather = df_weather.interpolate(method='pad', axis=0).ffill().bfill()
	
	# ============================
	# ========== Train ===========
	# ============================
	path = 'Data/ENTSOE_Energy_Load/Processed/train_details_' + str(hour_diff) + 'h_in' + str(input_size) + '.npy'
	train_details = np.load(path)
	print(train_details.shape)

	# Calendar
	train_details = pd.DataFrame(train_details[:, :2], columns=['Country', 'Datetime'])
	train_details['Datetime'] = pd.to_datetime(train_details['Datetime'])
	train_details['Date'] = train_details['Datetime'].dt.date

	train_details['temp_mean'] = 0
	train_details['temp_min'] = 0
	train_details['temp_max'] = 0

	for i in tqdm(range(train_details.shape[0])):
		# Key date
		key_date = train_details.iloc[i, :]['Date']
		key_country = train_details.iloc[i, :]['Country']

		# Extract country's temperatures
		df_temp = df_weather.loc[df_weather['Date']==key_date]
		t_mean = float(df_temp[key_country + '_mean'])
		t_min = float(df_temp[key_country + '_min'])
		t_max = float(df_temp[key_country + '_max'])

		# Change values
		train_details.iloc[i, 3] = t_mean
		train_details.iloc[i, 4] = t_min
		train_details.iloc[i, 5] = t_max

	train_details['mean_dist'] = np.abs(train_details['temp_mean'] - 18)
	train_details['min_dist'] = np.abs(train_details['temp_min'] - 18)
	train_details['max_dist'] = np.abs(train_details['temp_max'] - 18)
	train_details = train_details.drop(columns=['Datetime', 'Country', 'Date'])

	train_details = np.asarray(train_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_weather_' + str(hour_diff) + 'h_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, train_details)
	print(train_details.shape)

	# ============================
	# ========== Test ===========
	# ============================
	path = 'Data/ENTSOE_Energy_Load/Processed/x_test_in' + str(input_size) + '.npy'
	test_details = np.load(path)
	test_details = pd.DataFrame(test_details[:, 1:3], columns=['Country', 'Datetime'])
	test_details['Datetime'] = pd.to_datetime(test_details['Datetime'])
	test_details['Date'] = test_details['Datetime'].dt.date

	test_details['temp_mean'] = 0
	test_details['temp_min'] = 0
	test_details['temp_max'] = 0

	for i in tqdm(range(test_details.shape[0])):
		# Key date
		key_date = test_details.iloc[i, :]['Date']
		key_country = test_details.iloc[i, :]['Country']

		# Extract country's temperatures
		df_temp = df_weather.loc[df_weather['Date']==key_date]
		t_mean = float(df_temp[key_country + '_mean'])
		t_min = float(df_temp[key_country + '_min'])
		t_max = float(df_temp[key_country + '_max'])

		# Change values
		test_details.iloc[i, 3] = t_mean
		test_details.iloc[i, 4] = t_min
		test_details.iloc[i, 5] = t_max

	test_details['mean_dist'] = np.abs(test_details['temp_mean'] - 18)
	test_details['min_dist'] = np.abs(test_details['temp_min'] - 18)
	test_details['max_dist'] = np.abs(test_details['temp_max'] - 18)
	test_details = test_details.drop(columns=['Datetime', 'Country', 'Date'])


	test_details = np.asarray(test_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/x_test_weather_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, test_details)
	print(test_details.shape)
	print(np.min(test_details))
	print(np.max(test_details))


def exogenous_temperatures_weekdiff():
	# Params
	output_size = 36
	input_size = 168
	hour_diff = 12

	# Weather data 
	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_WeatherData.csv'
	df_weather = pd.read_csv(path)
	df_weather['Datetime'] = pd.to_datetime(df_weather['Unnamed: 0'])
	df_weather['Date'] = df_weather['Datetime'].dt.date
	df_weather = df_weather.interpolate(method='pad', axis=0).ffill().bfill()
	
	# ============================
	# ========== Train ===========
	# ============================
	path = 'Data/ENTSOE_Energy_Load/Processed/train_details_' + str(hour_diff) + 'h_in' + str(input_size) + '.npy'
	train_details = np.load(path)
	print(train_details.shape)

	# Calendar
	train_details = pd.DataFrame(train_details[:, :2], columns=['Country', 'Datetime'])
	train_details['Datetime'] = pd.to_datetime(train_details['Datetime'])
	train_details['Date'] = train_details['Datetime'].dt.date

	train_details['temp_mean'] = 0
	train_details['temp_min'] = 0
	train_details['temp_max'] = 0

	for i in tqdm(range(train_details.shape[0])):
		# Key date
		key_date = train_details.iloc[i, :]['Date']
		key_country = train_details.iloc[i, :]['Country']

		# Extract country's temperatures for the day of interest
		df_temp = df_weather.loc[df_weather['Date']==key_date]
		t_mean = float(df_temp[key_country + '_mean'])
		t_min = float(df_temp[key_country + '_min'])
		t_max = float(df_temp[key_country + '_max'])

		# Extract country's temperatures for the previous week
		idx = df_temp.index[0]
		df_temp_week = df_weather.iloc[(idx-7):idx]
		w_mean = float(df_temp_week[key_country + '_mean'].mean())
		w_min = float(df_temp_week[key_country + '_min'].mean())
		w_max = float(df_temp_week[key_country + '_max'].mean())

		# Change values
		train_details.iloc[i, 3] = t_mean - w_mean
		train_details.iloc[i, 4] = t_min - w_min
		train_details.iloc[i, 5] = t_max - w_max

	train_details['mean_dist'] = np.abs(train_details['temp_mean'] - 18)
	train_details['min_dist'] = np.abs(train_details['temp_min'] - 18)
	train_details['max_dist'] = np.abs(train_details['temp_max'] - 18)
	train_details = train_details.drop(columns=['Datetime', 'Country', 'Date'])

	train_details = np.asarray(train_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_weather_weekdiff_' + str(hour_diff) + 'h_in' + str(input_size) + '.npy'
	np.save(export_path, train_details)
	print(train_details.shape)

	# ============================
	# ========== Test ===========
	# ============================
	path = 'Data/ENTSOE_Energy_Load/Processed/x_test_in' + str(input_size) + '.npy'
	test_details = np.load(path)
	test_details = pd.DataFrame(test_details[:, 1:3], columns=['Country', 'Datetime'])
	test_details['Datetime'] = pd.to_datetime(test_details['Datetime'])
	test_details['Date'] = test_details['Datetime'].dt.date

	test_details['temp_mean'] = 0
	test_details['temp_min'] = 0
	test_details['temp_max'] = 0

	for i in tqdm(range(test_details.shape[0])):
		# Key date
		key_date = test_details.iloc[i, :]['Date']
		key_country = test_details.iloc[i, :]['Country']

		# Extract country's temperatures for the day of interest
		df_temp = df_weather.loc[df_weather['Date']==key_date]
		t_mean = float(df_temp[key_country + '_mean'])
		t_min = float(df_temp[key_country + '_min'])
		t_max = float(df_temp[key_country + '_max'])

		# Extract country's temperatures for the previous week
		idx = df_temp.index[0]
		df_temp_week = df_weather.iloc[(idx-7):idx]
		w_mean = float(df_temp_week[key_country + '_mean'].mean())
		w_min = float(df_temp_week[key_country + '_min'].mean())
		w_max = float(df_temp_week[key_country + '_max'].mean())

		# Change values
		test_details.iloc[i, 3] = t_mean - w_mean
		test_details.iloc[i, 4] = t_min - w_min
		test_details.iloc[i, 5] = t_max - w_max

	test_details['mean_dist'] = np.abs(test_details['temp_mean'] - 18)
	test_details['min_dist'] = np.abs(test_details['temp_min'] - 18)
	test_details['max_dist'] = np.abs(test_details['temp_max'] - 18)
	test_details = test_details.drop(columns=['Datetime', 'Country', 'Date'])

	test_details = np.asarray(test_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/x_test_weather_weekdiff_in' + str(input_size) + '.npy'
	np.save(export_path, test_details)
	print(test_details.shape)
	print(np.min(test_details))
	print(np.max(test_details))


def exogenous_temperatures_counorm():
	# Weather data processing
	path = 'Data/ENTSOE_Energy_Load/EnergyLoad_WeatherData.csv'
	df_weather = pd.read_csv(path)
	df_weather['Datetime'] = pd.to_datetime(df_weather['Unnamed: 0'])
	df_weather['Date'] = df_weather['Datetime'].dt.date
	df_weather = df_weather.interpolate(method='pad', axis=0).ffill().bfill()

	countries_list = df_weather.columns[1:-2]
	countries_list = list(set([x.split('_')[0] for x in countries_list]))
	proc_list = []
	for country in tqdm(countries_list):
		df_c = df_weather[['Datetime', 'Date']]
		df_c['Country'] = country

		df_temp = df_weather[['Date', country + '_mean', country + '_min', country + '_max']]
		df_c = df_c.merge(df_temp, on='Date', how='left')
		df_c = df_c.drop(columns='Datetime')
		df_c.columns = ['Date', 'Country', 'mean', 'min', 'max']

		df_c['mean_dist'] = np.abs(df_c['mean'] - 18)
		df_c['min_dist'] = np.abs(df_c['min'] - 18)
		df_c['max_dist'] = np.abs(df_c['max'] - 18)

		df_c['mean'] = (df_c['mean'] - df_c['mean'].mean()) / df_c['mean'].std()
		df_c['min'] = (df_c['min'] - df_c['min'].mean()) / df_c['min'].std()
		df_c['max'] = (df_c['max'] - df_c['max'].mean()) / df_c['max'].std()
		df_c['mean_dist'] = (df_c['mean_dist'] - df_c['mean_dist'].mean()) / df_c['mean_dist'].std()
		df_c['min_dist'] = (df_c['min_dist'] - df_c['min_dist'].mean()) / df_c['min_dist'].std()
		df_c['max_dist'] = (df_c['max_dist'] - df_c['max_dist'].mean()) / df_c['max_dist'].std()

		df_c = df_c.sort_values(by='Date', ascending=True, inplace=False)
		proc_list.append(df_c)

	df_weather = pd.concat(proc_list)

	# Params
	output_size = 36
	input_size = 168
	hour_diff = 24
	
	# ============================
	# ========== Train ===========
	# ============================
	path = 'Data/ENTSOE_Energy_Load/Processed/train_details_' + str(hour_diff) + 'h_in' + str(input_size) + '_counorm.npy'
	train_details = np.load(path)
	print(train_details.shape)

	# Calendar
	train_details = pd.DataFrame(train_details[:, :2], columns=['Country', 'Datetime'])
	train_details['Datetime'] = pd.to_datetime(train_details['Datetime'])
	train_details['Date'] = train_details['Datetime'].dt.date

	train_details['temp_mean'] = 0
	train_details['temp_min'] = 0
	train_details['temp_max'] = 0
	train_details['mean_dist'] = 0
	train_details['min_dist'] = 0
	train_details['max_dist'] = 0
	for i in tqdm(range(train_details.shape[0])):
		# Key date
		key_date = train_details.iloc[i, :]['Date']
		key_country = train_details.iloc[i, :]['Country']

		# Extract country's temperatures
		df_temp = df_weather.loc[(df_weather['Date']==key_date) & (df_weather['Country']==key_country)]
		
		# print('\n\n---------------')
		# print(key_country)
		# print(df_temp)
		# print(df_temp)
		# print(train_details.iloc[i, :])
		# t_mean = float(df_temp['mean'])
		# t_min = float(df_temp['min'])
		# t_max = 1

		# Change values
		train_details.iloc[i, 3] = float(df_temp['mean'])
		train_details.iloc[i, 4] = float(df_temp['min'])
		train_details.iloc[i, 5] = float(df_temp['max'])
		train_details.iloc[i, 6] = float(df_temp['mean_dist'])
		train_details.iloc[i, 7] = float(df_temp['min_dist'])
		train_details.iloc[i, 8] = float(df_temp['max_dist'])

		# print(train_details.iloc[i, :])
		# print('---------------\n\n')
		# print(stop)



	# train_details['mean_dist'] = np.abs(train_details['temp_mean'] - 18)
	# train_details['min_dist'] = np.abs(train_details['temp_min'] - 18)
	# train_details['max_dist'] = np.abs(train_details['temp_max'] - 18)
	train_details = train_details.drop(columns=['Datetime', 'Country', 'Date'])

	train_details = np.asarray(train_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/train_weather_' + str(hour_diff) + 'h_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, train_details)
	print(train_details.shape)


	# ============================
	# ========== Test ===========
	# ============================
	path = 'Data/ENTSOE_Energy_Load/Processed/x_test_in' + str(input_size) + '.npy'
	test_details = np.load(path)
	test_details = pd.DataFrame(test_details[:, 1:3], columns=['Country', 'Datetime'])
	test_details['Datetime'] = pd.to_datetime(test_details['Datetime'])
	test_details['Date'] = test_details['Datetime'].dt.date

	test_details['temp_mean'] = 0
	test_details['temp_min'] = 0
	test_details['temp_max'] = 0
	test_details['mean_dist'] = 0
	test_details['min_dist'] = 0
	test_details['max_dist'] = 0

	for i in tqdm(range(test_details.shape[0])):
		# Key date
		key_date = test_details.iloc[i, :]['Date']
		key_country = test_details.iloc[i, :]['Country']

		# Extract country's temperatures
		df_temp = df_weather.loc[(df_weather['Date']==key_date) & (df_weather['Country']==key_country)]

		# t_mean = float(df_temp[key_country + '_mean'])
		# t_min = float(df_temp[key_country + '_min'])
		# t_max = float(df_temp[key_country + '_max'])

		# # Change values
		# test_details.iloc[i, 3] = t_mean
		# test_details.iloc[i, 4] = t_min
		# test_details.iloc[i, 5] = t_max

		# Change values
		test_details.iloc[i, 3] = float(df_temp['mean'])
		test_details.iloc[i, 4] = float(df_temp['min'])
		test_details.iloc[i, 5] = float(df_temp['max'])
		test_details.iloc[i, 6] = float(df_temp['mean_dist'])
		test_details.iloc[i, 7] = float(df_temp['min_dist'])
		test_details.iloc[i, 8] = float(df_temp['max_dist'])


	# test_details['mean_dist'] = np.abs(test_details['temp_mean'] - 18)
	# test_details['min_dist'] = np.abs(test_details['temp_min'] - 18)
	# test_details['max_dist'] = np.abs(test_details['temp_max'] - 18)
	test_details = test_details.drop(columns=['Datetime', 'Country', 'Date'])


	test_details = np.asarray(test_details)
	export_path = 'Data/ENTSOE_Energy_Load/Processed/x_test_weather_in' + str(input_size) + '_counorm.npy'
	np.save(export_path, test_details)
	print(test_details.shape)
	print(np.min(test_details))
	print(np.max(test_details))


train_data_normalize_per_country()
test_data_normalize_per_country()
