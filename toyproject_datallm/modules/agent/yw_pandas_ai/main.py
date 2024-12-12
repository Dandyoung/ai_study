import pandas as pd
from pipeline import Pipeline
from split_dataframe_by_minutes import SplitDataFrameByMinutes
from aggregate_statistics import AggregateStatistics

def main():
    # 예제 데이터 프레임 생성
    # data = {
    #     'timestamp': pd.date_range(start='1/1/2022', periods=100, freq='T'),
    #     'value': range(100)
    # }
    # df = pd.DataFrame(data)

    df = pd.read_csv('/workspace/youngwoo/toyproject-datallm/modules/agent/tmp_dataset/3월24일.csv')

    # 첫 번째 열의 이름을 'timestamp'로 변경
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

    # 파이프라인 설정
    pipeline = Pipeline(steps=[
        SplitDataFrameByMinutes(minutes=4),
        AggregateStatistics()
    ])

    # 파이프라인 실행
    result = pipeline.run(df)
    for idx, stats in enumerate(result):
        print(f"Chunk {idx + 1} Statistics: {stats}")

if __name__ == "__main__":
    main()