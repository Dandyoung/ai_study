# 한글 폰트 설정 - 다른 방식으로 시도
import matplotlib as mpl
import matplotlib.font_manager as fm

mpl.rc('font', family='DejaVu Sans')  # 기본적으로 설치되어 있는 폰트

# 폰트 확인
print("사용 가능한 폰트:")
for font in fm.fontManager.ttflist:
    print(font.name)