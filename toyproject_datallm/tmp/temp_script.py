import pandas as pd
import json
df = pd.read_csv('temp_dataframe.csv')
result = df['1FY1902'].resample('1min').agg(['min', 'max', 'mean'])['2023-03-24']
with open('/workspace/youngwoo/toyproject-datallm/tmp/temp_result.json', 'w') as result_file:
    json.dump(result.to_dict(), result_file, ensure_ascii=False, indent=4)
