# 라이브러리 설정
import os
import pandas as pd
import numpy as np
from datetime import datetime
from datasets import load_dataset, Dataset, load_from_disk

# 평가 지표 라이브러리
import evaluate
from evaluate import load
import torch

# mlflow 라이브러리
import mlflow
import mlflow.pytorch

# transformers 라이브러리
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, pipeline   

# airflow 라이브러리
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

# 환경 변수 설정
os.environ['NO_PROXY'] = '*'  # airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요
# 모델 준비
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model_name = 'huawei-noah/TinyBERT_General_4L_312D'

# 모델과 데이터 경로. os.getcwd() 현재 작업 디렉토리 반환
# os.getenv는 환경 변수 가져오는 함수, os.path.join은 디렉토리 경로 합치는 함수
data_path = os.getenv("DATA_PATH", './data/IMDB_Dataset.csv')  # Use environment variable for flexibility
dataset_path = os.getenv("DATASET_PATH", './data/tk_dataset')
model_path = os.getenv("MODEL_PATH", './tinybert_model_test')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# MLflow 환경 설정
# mlflow.set_tracking_uri('http://host.docker.internal:5001')
# mlflow.set_tracking_uri('http://127.0.0.1:5000')
# mlflow.set_experiment('IMDB_Model_Training')

# Airflow 기본 설정
default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 11, 22),
    'retries': 1,
}

def load_tokenizer():
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def compute_accuracy(predictions):
    preds = np.argmax(predictions.predictions, axis=1)
    accuracy = evaluate.load('accuracy')
    return accuracy.compute(predictions=preds, 
                           references=predictions.label_ids)

# 데이터 전처리 함수 정의
def prepare_data(**kwargs):
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    
    # 라벨 딕셔너리 생성
    label2id = {'positive': 1, 'negative': 0}
    dataset = dataset.map(lambda x: {'label': label2id[x['sentiment']]})
    # dataset = dataset.map(tokenize_data, batched=True)

    # 토크나이저 적용
    tokenizer = load_tokenizer()
    dataset = dataset.map(
        lambda x: tokenizer(
            x['review'], 
            truncation=True, 
            max_length=300, 
            return_tensors='pt'
        ), 
        batched=True
    )
    dataset.save_to_disk(dataset_path)
    kwargs['ti'].xcom_push(key='label2id', value=label2id)

# 모델 학습 및 평가 함수 정의
def train_and_evaluate(**kwargs):
    ti = kwargs['ti']
    mlflow.set_tracking_uri('http://127.0.0.1:5001')
    mlflow.set_experiment('IMDB_Model_Training')
    
    label2id = ti.xcom_pull(key='label2id')
    id2label = {v: k for k, v in label2id.items()}
    dataset = Dataset.load_from_disk(dataset_path)
    

    # 모델 불러오기
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    # 학습 인자 설정
    args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy='epoch',
        learning_rate=2e-5
    )
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy    
    )

    mlflow.autolog()
    with mlflow.start_run(run_name=model_name):
        # 모델 학습 및 평가
        trainer.train()
        # 평가
        evaluation = trainer.evaluate()
        mlflow.log_metrics(evaluation)

        # 예측
        predictions = trainer.predict(dataset['test'])
        accuracy_score = compute_accuracy(predictions)
        mlflow.log_metrics({"test_accuracy": accuracy_score['accuracy']})

        
        # # Log hyperparameters
        # mlflow.log_param("num_train_epochs", args.num_train_epochs)
        # mlflow.log_param("learning_rate", args.learning_rate)
        # mlflow.log_param("batch_size", args.per_device_train_batch_size)

        trainer.save_model(model_path)
        mlflow.pytorch.log_model(model, artifact_path="model")

def slack_notification(context, message):
    SlackWebhookOperator(
        task_id='slack_notification_success',
        webhook_token = '***',  # slack webhook url 입력
        message=message,
        dag=context['dag']
    ).execute(context)

# DAG 정의
dag = DAG(
    'tinybert_ml_air',
    default_args=default_args,
    description='A machine learning pipeline with MLflow logging on IMDB dataset',
    schedule_interval='@daily',
    catchup=False
)

# Task 정의
prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    op_kwargs={},
    provide_context=True,
    dag=dag,
)
train_and_evaluate_task = PythonOperator(
    task_id='train_and_evaluate',
    python_callable=train_and_evaluate,
    op_kwargs={},
    provide_context=True,
    dag=dag,
)
slack_notification_task = SlackWebhookOperator(
    task_id='slack_notification',
    slack_webhook_conn_id='slack_webhook',
    message='Model Training Completed',
    dag=dag,
)

## Task 의존성 설정
prepare_data_task >> train_and_evaluate_task >> slack_notification_task

# if __name__ == "__main__":
#     # model_name = 'huawei-noah/TinyBERT_General_4L_312D'
#     with mlflow.start_run(run_name=model_name):
#         dag.cli()
