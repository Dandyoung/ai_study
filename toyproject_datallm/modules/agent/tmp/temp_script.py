# import pandas as pd
# df = pd.read_csv('/workspace/youngwoo/toyproject-datallm/modules/agent/tmp/temp_dataframe.csv')
# import pandas as pd
# import matplotlib.pyplot as plt

# # Assuming df is already loaded with the data for March 24th
# print(df.describe())

# # Plotting some trends
# plt.figure(figsize=(14, 7))
# plt.plot(df['datetime'], df['GT GEN#1 KV'], label='GT GEN#1 KV')
# plt.plot(df['datetime'], df['GT GEN#1 KVOLT'], label='GT GEN#1 KVOLT')
# plt.plot(df['datetime'], df['GT GEN#1 MVAR'], label='GT GEN#1 MVAR')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.title('Trends for March 24th')
# plt.legend()
# plt.grid(True)
# plt.show()

import pandas as pd

# Sample data creation (assuming the dataframe is named 'df')
data = {
    'datetime': ['2023-03-24 00:00:00', '2023-03-24 00:00:01', '2023-03-24 00:00:02', '2023-03-24 00:00:03', '2023-03-24 00:00:04'],
    'GT GEN#1 KV': [17.8628, 17.8669, 17.873, 17.8635, 17.8641],
    'GT GEN#1 KVOLT': [17.7139, 17.7172, 17.7217, 17.7263, 17.7299],
    'GT GEN#1 MVAR': [15.0218, 15.5759, 15.726, 15.3231, 15.0894]
}
df = pd.DataFrame(data)
df['datetime'] = pd.to_datetime(df['datetime'])

# Set 'datetime' as the index
df.set_index('datetime', inplace=True)

# Resample the data to minute intervals
resampled_data = df.resample('T').agg(['mean', 'min', 'max', 'std'])

# Display the resampled data
print(resampled_data)

