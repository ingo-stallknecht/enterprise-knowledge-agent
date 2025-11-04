# scripts/eval_rag.py
import json, yaml
from app.rag.retriever import Retriever
from app.rag.evaluator import SimpleEvaluator
import mlflow


cfg = yaml.safe_load(open("configs/settings.yaml"))
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment"])


retriever = Retriever(
cfg["retrieval"]["embedder_model"],
cfg["retrieval"]["normalize"],
cfg["retrieval"]["faiss_index"],
cfg["retrieval"]["store_json"],
)


questions = [json.loads(l) for l in open(cfg["eval"]["questions_path"]) if l.strip()]
evalr = SimpleEvaluator(retriever, cfg["eval"]["k"])
metrics = evalr.eval_questions(questions)
print("Eval:", metrics)


with mlflow.start_run() as run:
for k, v in metrics.items():
mlflow.log_metric(k, v)
mlflow.log_param("embedder", cfg["retrieval"]["embedder_model"])
mlflow.log_artifact(cfg["eval"]["questions_path"], artifact_path="eval")
mlflow.set_tag("index_path", cfg["retrieval"]["faiss_index"])
mlflow.set_tag("store_path", cfg["retrieval"]["store_json"])
mlflow.set_tag("stage_candidate", "true")
print("Run:", run.info.run_id)