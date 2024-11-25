모델 및 서비스 선택과 방법론 기록

## 데이터 설명
### IMDB 영화 리뷰
### 한국어 영화 리뷰

## 트랜스포머 모델
### BERT/ALBERT
### 한국어 트랜스포너 

## FAST API

## Airflow 구조


## 주의 사항 
- 컨테이너 접속 후 mlflow ui 명령어로 mlflow 서버 실행
    - Volumes 에도 mlflow 폴더를 연동해줘야 하나요? => 작업공간에 세팅 (호스트에서 보고 싶으시면 그렇게 하면됩니다.)
    - 작업공간에 세팅하려면 Dockerfile 수정 필요. 작업공간: WORKDIR 
