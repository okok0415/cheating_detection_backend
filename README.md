1. git clone https://github.com/okok0415/cheating_detection_backend.git

2. 파일 접속 후 python -m venv venv (가상환경 생성)

3. venv/Scripts/activate (cmd), venv/Scripts/Activate.ps1(powershell) (가상환경 들어가기 프로젝트 실행할 때 마다 실행해야됨)

4. pip install -r requirements.txt

5. https://drive.google.com/file/d/1bTyrowPFUbGzzs3ZrfEglp1wfMs1JnJx/view?usp=sharing
링크 따라 들어가서 .deepface, Tesseract-OCR, cvdata 다운로드 후 cheating_detection_backend 파일안에 넣어준다.

6. python manage.py runserver - 서버 실행

7. POSTMAN 사용하면 frontend없이 데이터 보내는 것 가능

https://drive.google.com/file/d/1bTyrowPFUbGzzs3ZrfEglp1wfMs1JnJx/view?usp=sharing