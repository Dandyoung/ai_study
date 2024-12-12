import pandas as pd

def create_info_df():
    # 태그 이름, 설명, 샘플값 리스트
    # tags = [
    #     "datetime", "1FY1902", "1GT1KV", "1GT1MVAR", "1GT1MWTRGT", "1GT1MWTRGT-11",
    #     "1V.GGG.EGA.A", "1V.GJG.EGA.A", "1V.GJG.MGA.A", "1V.GJG.PGA.A", 
    #     "1V.GJG.PGA.D", "1V.GXG.LGA.A", "1V.GXG.PGA.A", "1V.IGJ.A0N.A", 
    #     "1V.IGJ.A0O.A", "1V.IJG.AGA.A", "1V.IJG.GGA.A", "1V.IMU.AF0.A", 
    #     "1V.IPG.HGA.A", "1V.IPJ.AC1.A", "1V.IPJ.AC2.A", "1V.IPJ.ADD.A", 
    #     "1V.IPJ.AXD.A", "1V.IPJ.NGA.A", "1V.ITG.CGA.A", "1V.ITU.AF0.A", 
    #     "1V.ITU.AF1.A", "1JYR1918", "1JYS1919", "1JYT1945", "1GTDL1001", 
    #     "1PDIS1501A"
    # ]

    tags = [
    "datetime",
    "GT GEN#1 KV",
    "GT GEN#1 KVOLT",
    "GT GEN#1 MVAR",
    "GT GEN#1 MEGAWATT TARGET",
    "GT#1 Generator Efficiency",
    "GT#1 Generator Electronic Loss",
    "GT#1 Generator Mechanical Loss",
    "GT#1 Generator Loss by Power Factor Curve",
    "GT#1 Generator Loss by Power Factor Curve at Design Power Factor and Actual MW",
    "GT#1 Generator Exciter Loss by Curve",
    "GT#1 Generator Power Factor Actual",
    "GT#1 ppm NOx in Exhaust Gas",
    "GT#1 % O2 in Exhaust Flue Gas",
    "GT#1 Generator Auxiliary (MW)",
    "GT#1 Generator Gross Generation (MW)",
    "GT#1 Fuel Mass Flow",
    "GT#1 Generator Hydrogen Pressure",
    "GT#1 Compressor Inlet Pressure",
    "GT#1 Compressor Discharge Pressure",
    "GT#1 Inlet Pressure Loss",
    "GT#1 Turbine Exhaust Draft Loss",
    "GT#1 NG WTR Pressure",
    "GT#1 Generator Cold Gas Temperature",
    "GT#1 Fuel Temperature",
    "GT#1 Fuel Inlet Temperature",
    "G1 START",
    "G1 STOP",
    "G1 GT TRIP",
    "S-TBQC-035 In air tot press tr",
    "G1 FG HE1 I/L FLTR DIFF.PR HI"
    ]
    descriptions = [
    "datetime. 센서 데이터가 수집된 시간을 나타냅니다. 시간은 1초 단위로 기록됩니다.",
    "GT GEN#1 KV. 발전기 출력 전압을 나타냅니다. 이는 발전기에서 생성된 전압을 측정한 값입니다.",
    "GT GEN#1 KVOLT. 출력 전압을 나타냅니다. 이는 발전기에서 측정된 전압으로, 실제 전압값을 나타냅니다.",
    "GT GEN#1 MVAR. 무효 전력을 나타냅니다. 무효 전력은 전기 시스템에서 전압을 유지하기 위해 필요한 전력입니다.",
    "GT GEN#1 MEGAWATT TARGET. 목표 유효 전력을 나타냅니다. 이는 발전기의 목표 출력 전력입니다.",
    "GT#1 Generator Efficiency. 발전 효율을 나타냅니다. 이는 발전기의 전기적 효율을 백분율로 표시합니다.",
    "GT#1 Generator Electronic Loss. 전자 손실을 나타냅니다. 전력 변환 과정 중 발생하는 전자 손실을 의미합니다.",
    "GT#1 Generator Mechanical Loss. 기계적 손실을 나타냅니다. 발전기 기계적 부분에서 발생하는 손실을 의미합니다.",
    "GT#1 Generator Loss by Power Factor Curve. 역률 곡선에 따른 손실을 나타냅니다. 이는 역률에 따라 발생하는 손실을 의미합니다.",
    "GT#1 Generator Loss by Power Factor Curve at Design Power Factor and Actual MW. 설계된 역률과 실제 출력에 따른 손실을 나타냅니다. 이는 특정 조건에서 발생하는 손실을 계산합니다.",
    "GT#1 Generator Exciter Loss by Curve. 곡선에 따른 여자기 손실을 나타냅니다. 이는 여자기의 효율에 따른 손실을 의미합니다.",
    "GT#1 Generator Power Factor Actual. 실제 역률을 나타냅니다. 이는 실제 운전 조건에서의 발전기 역률을 의미합니다.",
    "GT#1 ppm NOx in Exhaust Gas. 배기가스 중 NOx 농도를 나타냅니다. 이는 배기가스에 포함된 질소 산화물의 농도를 ppm 단위로 나타냅니다.",
    "GT#1 % O2 in Exhaust Flue Gas. 배기가스 중 산소 비율을 나타냅니다. 이는 배기가스에 포함된 산소의 비율을 백분율로 나타냅니다.",
    "GT#1 Generator Auxiliary (MW). 보조 전력을 나타냅니다. 이는 발전기의 보조 시스템에서 소비되는 전력입니다.",
    "GT#1 Generator Gross Generation (MW). 총 발전량을 나타냅니다. 이는 발전기가 생성한 총 전력을 MW 단위로 나타냅니다.",
    "GT#1 Fuel Mass Flow. 연료 질량 유량을 나타냅니다. 이는 연료가 연소실로 유입되는 질량 유량을 kg/s 단위로 나타냅니다.",
    "GT#1 Generator Hydrogen Pressure. 수소 압력을 나타냅니다. 이는 발전기 냉각 시스템의 수소 압력을 의미합니다.",
    "GT#1 Compressor Inlet Pressure. 압축기 흡입 압력을 나타냅니다. 이는 압축기 입구에서의 압력을 나타냅니다.",
    "GT#1 Compressor Discharge Pressure. 압축기 배출 압력을 나타냅니다. 이는 압축기 출구에서의 압력을 나타냅니다.",
    "GT#1 Inlet Pressure Loss. 흡입 압력 손실을 나타냅니다. 이는 압축기 입구에서 발생하는 압력 손실을 의미합니다.",
    "GT#1 Turbine Exhaust Draft Loss. 터빈 배기 손실을 나타냅니다. 이는 터빈 배출구에서 발생하는 압력 손실을 의미합니다.",
    "GT#1 NG WTR Pressure. 천연가스 물 압력을 나타냅니다. 이는 천연가스 공급 시스템의 압력을 나타냅니다.",
    "GT#1 Generator Cold Gas Temperature. 냉각 가스 온도를 나타냅니다. 이는 발전기 냉각 가스의 온도를 의미합니다.",
    "GT#1 Fuel Temperature. 연료 온도를 나타냅니다. 이는 연료의 온도를 의미합니다.",
    "GT#1 Fuel Inlet Temperature. 연료 흡입 온도를 나타냅니다. 이는 연료가 연소실로 들어가기 전의 온도를 의미합니다.",
    "G1 START. 시작 명령을 나타냅니다. 이는 시작 상태를 의미합니다.",
    "G1 STOP. 정지 명령을 나타냅니다. 이는 정지 상태를 의미합니다.",
    "G1 GT TRIP. 비상 정지를 나타냅니다. 이는 비상 정지 상태를 의미합니다.",
    "S-TBQC-035 In air tot press tr. 공기 전체 압력 트랜스미터를 나타냅니다. 이는 공기 압력을 측정하는 장치를 의미합니다.",
    "G1 FG HE1 I/L FLTR DIFF.PR HI. 연료 가스 히터 입구 필터 차압 높음을 나타냅니다. 이는 연료 가스 히터 입구 필터에서의 차압이 높을 때 발생하는 경고를 의미합니다."
    ]
    # descriptions = [
    #     "datetime. 센서 데이터가 수집된 시간을 나타냅니다. 시간은 1초 단위로 기록됩니다.",
    #     "GT GEN#1 KV. 가스터빈 1호기의 발전기 출력 전압을 나타냅니다. 이는 발전기에서 생성된 전압을 측정한 값입니다.",
    #     "GT GEN#1 KVOLT. 가스터빈 1호기의 출력 전압을 나타냅니다. 이는 발전기에서 측정된 전압으로, 실제 전압값을 나타냅니다.",
    #     "GT GEN#1 MVAR. 가스터빈 1호기의 무효 전력을 나타냅니다. 무효 전력은 전기 시스템에서 전압을 유지하기 위해 필요한 전력입니다.",
    #     "GT GEN#1 MEGAWATT TARGET. 가스터빈 1호기의 목표 유효 전력을 나타냅니다. 이는 가스터빈 발전기의 목표 출력 전력입니다.",
    #     "GT GEN#1 MEGAWATT TARGET. 가스터빈 1호기의 목표 유효 전력을 나타냅니다. 이는 발전기의 설정된 목표 출력 전력을 나타냅니다.",
    #     "GT#1 Generator Efficiency. 가스터빈 1호기의 발전 효율을 나타냅니다. 이는 발전기의 전기적 효율을 백분율로 표시합니다.",
    #     "GT#1 Generator Electronic Loss. 가스터빈 1호기의 전자 손실을 나타냅니다. 전력 변환 과정 중 발생하는 전자 손실을 의미합니다.",
    #     "GT#1 Generator Mechanical Loss. 가스터빈 1호기의 기계적 손실을 나타냅니다. 발전기 기계적 부분에서 발생하는 손실을 의미합니다.",
    #     "GT#1 Generator Loss by Power Factor Curve. 역률 곡선에 따른 가스터빈 1호기의 손실을 나타냅니다. 이는 역률에 따라 발생하는 손실을 의미합니다.",
    #     "GT#1 Generator Loss by Power Factor Curve at Design Power Factor and Actual MW. 설계된 역률과 실제 출력에 따른 가스터빈 1호기의 손실을 나타냅니다. 이는 특정 조건에서 발생하는 손실을 계산합니다.",
    #     "GT#1 Generator Exciter Loss by Curve. 곡선에 따른 가스터빈 1호기의 여자기 손실을 나타냅니다. 이는 여자기의 효율에 따른 손실을 의미합니다.",
    #     "GT#1 Generator Power Factor Actual. 가스터빈 1호기의 실제 역률을 나타냅니다. 이는 실제 운전 조건에서의 발전기 역률을 의미합니다.",
    #     "GT#1 ppm NOx in Exhaust Gas. 배기가스 중 NOx 농도를 나타냅니다. 이는 배기가스에 포함된 질소 산화물의 농도를 ppm 단위로 나타냅니다.",
    #     "GT#1 % O2 in Exhaust Flue Gas. 배기가스 중 산소 비율을 나타냅니다. 이는 배기가스에 포함된 산소의 비율을 백분율로 나타냅니다.",
    #     "GT#1 Generator Auxiliary (MW). 가스터빈 1호기의 보조 전력을 나타냅니다. 이는 발전기의 보조 시스템에서 소비되는 전력입니다.",
    #     "GT#1 Generator Gross Generation (MW). 가스터빈 1호기의 총 발전량을 나타냅니다. 이는 발전기가 생성한 총 전력을 MW 단위로 나타냅니다.",
    #     "GT#1 Fuel Mass Flow. 가스터빈 1호기의 연료 질량 유량을 나타냅니다. 이는 연료가 연소실로 유입되는 질량 유량을 kg/s 단위로 나타냅니다.",
    #     "GT#1 Generator Hydrogen Pressure. 가스터빈 1호기의 수소 압력을 나타냅니다. 이는 발전기 냉각 시스템의 수소 압력을 의미합니다.",
    #     "GT#1 Compressor Inlet Pressure. 가스터빈 1호기의 압축기 흡입 압력을 나타냅니다. 이는 압축기 입구에서의 압력을 나타냅니다.",
    #     "GT#1 Compressor Discharge Pressure. 가스터빈 1호기의 압축기 배출 압력을 나타냅니다. 이는 압축기 출구에서의 압력을 나타냅니다.",
    #     "GT#1 Inlet Pressure Loss. 가스터빈 1호기의 흡입 압력 손실을 나타냅니다. 이는 압축기 입구에서 발생하는 압력 손실을 의미합니다.",
    #     "GT#1 Turbine Exhaust Draft Loss. 가스터빈 1호기의 터빈 배기 손실을 나타냅니다. 이는 터빈 배출구에서 발생하는 압력 손실을 의미합니다.",
    #     "GT#1 NG WTR Pressure. 천연가스 물 압력을 나타냅니다. 이는 천연가스 공급 시스템의 압력을 나타냅니다.",
    #     "GT#1 Generator Cold Gas Temperature. 가스터빈 1호기의 냉각 가스 온도를 나타냅니다. 이는 발전기 냉각 가스의 온도를 의미합니다.",
    #     "GT#1 Fuel Temperature. 가스터빈 1호기의 연료 온도를 나타냅니다. 이는 연료의 온도를 의미합니다.",
    #     "GT#1 Fuel Inlet Temperature. 가스터빈 1호기의 연료 흡입 온도를 나타냅니다. 이는 연료가 연소실로 들어가기 전의 온도를 의미합니다.",
    #     "G1 START. 가스터빈 1호기의 시작 명령을 나타냅니다. 이는 가스터빈 1호기의 시작 상태를 의미합니다.",
    #     "G1 STOP. 가스터빈 1호기의 정지 명령을 나타냅니다. 이는 가스터빈 1호기의 정지 상태를 의미합니다.",
    #     "G1 GT TRIP. 가스터빈 1호기의 비상 정지를 나타냅니다. 이는 가스터빈 1호기의 비상 정지 상태를 의미합니다.",
    #     "S-TBQC-035 In air tot press tr. 공기 전체 압력 트랜스미터를 나타냅니다. 이는 공기 압력을 측정하는 장치를 의미합니다.",
    #     "G1 FG HE1 I/L FLTR DIFF.PR HI. 연료 가스 히터 입구 필터 차압 높음. 이는 연료 가스 히터 입구 필터에서의 차압이 높을 때 발생하는 경고를 의미합니다."
    # ]

    # sample_values = [
    #     "2023-07-09 11:57:24", 17.86278152, 17.71389008, 15.02184486, 120, 99.14781952, 0, 0, 813.0512695, 
    #     959.4055176, 0, 0.992569447, 6.716916084, 13.93929005, 2.362693787, 119.3618774, 26404.77344, 2.230944157, 
    #     "Pt Created", 12.49079227, 0.005811641, 0.025218034, 0.084749617, 30.21399498, 184.2317963, 10.66026974, 
    #     "OFF", "OFF", "OFF", 0, "ON"
    # ]

    sample_types = [
        "datetime", "float64", "float64", "float64", "float64", "float64", "int64", "int64", "float64",
        "float64", "int64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "object",
        "float64", "float64", "float64", "float64", "float64", "float64", "float64", "object", "object", "object",
        "int64", "object"
    ]

    category = [
    "시간", "전압", "전압", "전력", "전력", "효율성", "손실", "손실", "손실", "손실", 
    "손실", "손실", "배기가스", "배기가스", "전력", "전력", "연료", "압력", "압력", 
    "압력", "손실", "손실", "압력", "온도", "온도", "온도", "상태", "상태", "상태", "압력", "압력"
    ]

    # 데이터프레임 생성
    info_df = pd.DataFrame({
        "센서 태그명": tags,
        "센서 분류" : category,
        "센서 타입": sample_types,
        #"샘플": sample_values,
        "설명": descriptions
    })

    return info_df
