# Cheating Detection on Online Test

부정행위를 탐지할 수 있는 온라인 시험 플랫폼이다. 부정행위를 방지하기위해 Authentication 기능을 도입하였고, 시험 도중 카메라 밖의 사각지대에 있는 컨닝 페이퍼를 보는 것을 방지하기 위해 EyeTracking 기능을 넣었다.

## How to Use
1. git clone https://github.com/okok0415/cheating_detection_backend.git

2. 파일 접속 후 python -m venv venv (가상환경 생성)

3. venv/Scripts/activate (cmd), venv/Scripts/Activate.ps1(powershell) (가상환경 들어가기 프로젝트 실행할 때 마다 실행해야됨)

4. pip install -r requirements.txt 

    4.1 eos-py의 경우는 Visual Studio 2019에서 'Desktop development with C++'를 설치하고 진행
    
    4.2 Pytorch의 경우 사용자의 그래픽카드에 맞는 torch를 공식 홈페이지에서 다운받아 진행

5. https://drive.google.com/drive/folders/19mypcFLYxafz8EoJkTdxAOqwuBgO5Nxa?usp=sharing 에서 face_detector,models, Tesseract-OCR 다운로드 후 authentication 파일안에 넣어준다.

6. python manage.py makemigrations

7. python manage.py migrate 

8. python manage.py runserver - 서버 실행

9. 현재 mongoDB의 개인 DB포트에 연결하는 형태이다. mongoDB가 없다면 config/settings.py/ DATABASE를 default로 바꾼다. (기존 데이터 베이스 밑에 주석처리 해놓음)

10. Redis 다운로드 받으세요. github.com/tporadowski/redis/releases

11. (for windows) 하다가 gcc && cmake && visual studio && make && torch

# Release

|날짜|내용
|---|---|
|2021.11.05| 여러가지 기능(Authentication, Eye tracking)을 Merge 하였음




