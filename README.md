1. git clone https://github.com/okok0415/cheating_detection_backend.git

2. 파일 접속 후 python -m venv venv (가상환경 생성)

3. venv/Scripts/activate (cmd), venv/Scripts/Activate.ps1(powershell) (가상환경 들어가기 프로젝트 실행할 때 마다 실행해야됨)

4. pip install -r requirements.txt

5. https://drive.google.com/drive/folders/19mypcFLYxafz8EoJkTdxAOqwuBgO5Nxa?usp=sharing 에서 face_detector,models, Tesseract-OCR 다운로드 후 authentication 파일안에 넣어준다.

6. python manage.py makemigrations

7. python manage.py migrate 

8. python manage.py runserver - 서버 실행

9. 현재 mongoDB의 개인 DB포트에 연결하는 형태이다. mongoDB가 없다면 config/settings.py/ DATABASE를 default로 바꾼다. (기존 데이터 베이스 밑에 주석처리 해놓음)
