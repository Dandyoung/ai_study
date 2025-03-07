# Python 환경 설정 가이드

이 문서는 Windows와 Linux에서 Python 개발 환경을 설정하는 방법을 담고 있습니다. pyenv를 사용한 Python 버전 관리와 Poetry를 사용한 의존성 관리 방법을 다룹니다.

## 목차
- [Windows에서 pyenv 설치](#windows에서-pyenv-설치)
- [Linux에서 pyenv 설치](#linux에서-pyenv-설치)
- [pyenv로 Python 설치 및 관리](#pyenv로-python-설치-및-관리)
- [Poetry 사용 방법](#poetry-사용-방법)

## Windows에서 pyenv 설치

### pyenv-win 설치
```bash
git clone https://github.com/pyenv-win/pyenv-win.git "$env:USERPROFILE\.pyenv"
```

### 환경변수 설정
아래의 명령어를 PowerShell에서 실행:

```bash
[System.Environment]::SetEnvironmentVariable('PYENV', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
[System.Environment]::SetEnvironmentVariable('PYENV_ROOT', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
[System.Environment]::SetEnvironmentVariable('PYENV_HOME', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
```

PATH 환경변수 설정:

```bash
[System.Environment]::SetEnvironmentVariable('PATH', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('PATH', "User"), "User")
```

### 설치 확인
PowerShell을 재시작한 후 다음 명령어로 확인:

```bash
pyenv
```

## Linux에서 pyenv 설치

### 의존성 패키지 설치
Ubuntu/Debian 기준:

```bash
apt-get update
apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
libssl-dev
```

### pyenv 설치

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```

또는:

```bash
curl https://pyenv.run | bash
```

### 환경변수 설정
Bash 사용자 (~/.bashrc):

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

Zsh 사용자 (~/.zshrc):

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

### 설정 적용
셸 설정 적용:

```bash
source ~/.bashrc
```

또는:

```bash
exec $SHELL
```

### 설치 확인

```bash
pyenv --version
```

## pyenv로 Python 설치 및 관리

### Python 설치

```bash
pyenv install 3.11
```

### 전역 Python 버전 설정

```bash
pyenv global 3.11
```

### Python 버전 확인

```bash
python --version
```

### 유용한 pyenv 명령어

사용 가능한 Python 버전 목록 확인:
```bash
pyenv install --list
```

특정 프로젝트에만 Python 버전 지정(로컬 설정):
```bash
cd 프로젝트_디렉토리
pyenv local 3.11
```

설치된 Python 버전 목록 확인:
```bash
pyenv versions
```

## Poetry 사용 방법

### 가상환경 활성화 (Poetry 2.0.0 이상)

새로운 방식 (권장):
```bash
poetry env activate
```

또는 전체 경로 지정:
```bash
poetry env activate $(poetry env info --path)
```

Poetry는 가상환경을 생성하고 활성화 명령어를 출력합니다. 예시:
```bash
02:52:33 root@fb1117739110 Document ±|main ✗|→ poetry env activate
Creating virtualenv langchain-kr-qOi3KfN2-py3.11 in /root/.cache/pypoetry/virtualenvs
source /root/.cache/pypoetry/virtualenvs/langchain-kr-qOi3KfN2-py3.11/bin/activate
```

표시된 명령어를 직접 실행해야 합니다:
```bash
source /root/.cache/pypoetry/virtualenvs/langchain-kr-qOi3KfN2-py3.11/bin/activate
```

명령어 실행 후 프롬프트가 변경되어 가상환경이 활성화됨을 보여줍니다:
```bash
02:53:04 (langchain-kr-qOi3KfN2-py3.11) root@fb1117739110 Document ±|main ✗|→
```

### Shell 플러그인 설치 (선택 사항)
기존 `poetry shell` 명령을 계속 사용하려면:

```bash
poetry self add poetry-plugin-shell
```

### 가상환경 종료

```bash
deactivate
```

### 명령어 실행

가상환경 내에서 명령 실행:
```bash
poetry run [명령어]
```

---

이 가이드는 Python 개발 환경 설정의 기본적인 방법을 다루고 있습니다. 더 자세한 정보는 각 도구의 공식 문서를 참조하세요:
- [pyenv 공식 문서](https://github.com/pyenv/pyenv)
- [Poetry 공식 문서](https://python-poetry.org/docs/)