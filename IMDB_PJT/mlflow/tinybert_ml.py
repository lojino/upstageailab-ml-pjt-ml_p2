# 라이브러리 설정
import os
import pandas as pd
import numpy as np
from datetime import datetime
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

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

# # airflow 라이브러리
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator


# 환경 변수 설정
os.environ['NO_PROXY'] = '*'  # airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요
# 모델 준비
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model_name = 'huawei-noah/TinyBERT_General_4L_312D'
data_path = os.getenv('DATA_PATH', './data/IMDB_Dataset.csv')
dataset_path = os.getenv('DATASET_PATH', './data/tk_dataset')
model_path = os.getenv('MODEL_PATH', './tinybert_model_test')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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

def prepare_data():
    df = pd.read_csv(data_path, index_col=0)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    
    label2id = {'positive': 1, 'negative': 0}
    dataset = dataset.map(lambda x: {'label': label2id[x['sentiment']]})

    tokenizer = load_tokenizer()
    dataset = dataset.map(
        lambda x: tokenizer(
            x['review'], 
            padding='max_length', 
            truncation=True, 
            max_length=300, 
            return_tensors=None # mlflow만 사용할 시 None으로,  'pt'는 pytorch tensor로 반환하는 것을 의미
        ), 
        batched=True
    )
    dataset.save_to_disk(dataset_path)
    return label2id

def train_and_evaluate(label2id):
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('IMDB_Model_Training')

    id2label = {0: 'negative', 1: 'positive'}
    dataset = DatasetDict.load_from_disk(dataset_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy='epoch',
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy
    )
    trainer.train()
    evaluation = trainer.evaluate()
    mlflow.log_metrics(evaluation)

    predictions = trainer.predict(dataset['test'])
    accuracy_score = compute_accuracy(predictions)
    mlflow.log_metrics({"test_accuracy": accuracy_score['accuracy']})

    trainer.save_model(model_path)
    mlflow.pytorch.log_model(model, model_path)  # mlflow에 모델 저장 pytorch 형식



if __name__ == "__main__":
    # model_name = 'huawei-noah/TinyBERT_General_4L_312D'
    dataset, label2id = prepare_data()
    mlflow.autolog()
    with mlflow.start_run(run_name=model_name):
        train_and_evaluate(dataset, label2id)
