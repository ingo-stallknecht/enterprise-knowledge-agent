#!/usr/bin/env bash
mkdir -p mlflow_artifacts
mlflow server \
--backend-store-uri file:./mlflow_artifacts \
--default-artifact-root file:./mlflow_artifacts \
--host 127.0.0.1 --port 5000