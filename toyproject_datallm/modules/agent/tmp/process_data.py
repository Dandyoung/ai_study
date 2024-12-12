import pandas as pd

# 데이터 로드
df = pd.read_csv('data.csv')

# datetime 컬럼을 datetime 타입으로 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 분 단위로 데이터를 리샘플링
df.set_index('datetime', inplace=True)
df_resampled = df.resample('1T').mean()

# 기초 통계 정보 출력
df_stats = df_resampled.describe()

# 결과를 csv 파일로 저장
df_stats.to_csv('result.csv')