# dataLLM

AI팀 토이프로젝트 입니다. 

모든 파일이 구현된 것은 아니며, 필요에 따라 제외되거나 신규 추가될 수 있습니다. 

이는 프로젝트가 진행되면서 요구 사항에 맞춰 지속적으로 개선 및 조정이 이루어질 수 있음을 의미합니다.

![DataLLM Architecture](/img/architecture_ver3.PNG)

## 프로젝트 구조
```
llmchatflow/
├── .gitignore
├── README.md
├── requirements.txt
├── img/
│   └── architecture.jpg
├── notebook/
│   └── exel_to_sqlite.ipynb
├── modules/
│   ├── agent/
│   │   ├── __init__.py 
│   │   └── react_agent.py
│   │   └── react_agent_gpt_4.py
│   └── chains/
│       ├── __init__.py
│       └── create_sql_chain.py
├── prompt_templates/
│   ├── __init__.py
│   └── prompt.py
└── tests/
    └── __init__.py
```

## 프로젝트 구조

### `/img`
- 이미지 저장 폴더

### `/notebook`
- 테스트용 주피터 노트북 저장 폴더

### `/modules`
- `/agent`: agent 관련 모듈 폴더
  - `react_agent.py`: gpt-3.5 react zeroshot agent 모듈
  - `react_agent_gpt_4.py`: gpt-4 react zeroshot agent 모듈
- `/chains`: chain 관련 모듈 폴더
  - `create_sql_chain.py`: 자연어 -> sql 쿼리 변경 체인

### `/prompt_templates`
- `prompt.py`: 코드에 사용되는 모든 prompt를 관리할 폴더

### `/tests`
- 테스트용 python script 저장 폴더

# 브랜치 전략

프로젝트의 브랜치 전략은 다음과 같습니다. 각 브랜치의 역할과 기능을 명확히 이해하여 효율적인 협업과 코드 관리가 이루어질 수 있도록 합니다.

## 주요 브랜치

- **master 브랜치**
  - 안정적이고 배포 가능한 버전의 코드를 유지합니다.
  - 배포 준비가 완료된 코드만 master 브랜치에 병합됩니다.

## 지원 브랜치

- **feature 브랜치**
  - 특정 기능을 개발할 때 사용합니다.(chain / agent)