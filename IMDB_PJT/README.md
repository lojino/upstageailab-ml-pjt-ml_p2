모델 및 서비스 선택과 방법론 기록

## [데이터 설명]
### IMDB 영화 리뷰
### 한국어 영화 리뷰

## [MLflow]
### 전처리
### 트랜스포머 모델
- BERT/ALBERT
- 한국어 트랜스포머(KoBERT?)


## [Airflow 구조]
### Task 의존성
- *********
### Dag(기능) List
1. 영화 리뷰 추출 (imdb 등)
2. 추출 데이터 전저치 (토큰화 등)
3. 모델 train - - MLFlow 
4. 모델 evaluate - MLFlow 
5. 학습 모델 저장
6. 슬랙봇으로 진행 과정 알림 

## [모델 저장]
- AWS
- MLflow model registry
    - staging/production

## [FAST API/Streamlit]
- 앞에서 저장된 모델 활용 배포 
- Docker로 data input > 모델 > data output 자동화
- streamlit으로 web app 기능/디자인 구축

## [Slack Noti]
### 연결 
- slack webhook 설정 추가 
- airflow > admin > connections (url 가져오기)
### 기능 
- mlflow 학습/평가 결과
- airflow 진행 사항 모니터링
- 모델 파이프라인 에러 확인 


## 주의 사항 
- 컨테이너 접속 후 mlflow ui 명령어로 mlflow 서버 실행
    - Volumes 에도 mlflow 폴더를 연동해줘야 하나요? => 작업공간에 세팅 (호스트에서 보고 싶으시면 그렇게 하면됩니다.)
    - 작업공간에 세팅하려면 Dockerfile 수정 필요. 작업공간: WORKDIR 
