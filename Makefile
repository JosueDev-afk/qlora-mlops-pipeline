.DEFAULT_GOAL := help
# Docker or Podman, whichever is installed; override with ENGINE=docker|podman.
ENGINE   ?= $(shell command -v docker >/dev/null 2>&1 && echo docker || echo podman)
COMPOSE  := $(ENGINE) compose --env-file .env -f infra/docker-compose.yml
ALL      := --profile pipeline --profile streaming --profile bi --profile hdfs
PROFILES ?= pipeline
DAG      := $(COMPOSE) exec -T airflow-scheduler airflow dags trigger
# Host targets run in the project venv whether or not it is activated.
PYTHON   ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

# ── infrastructure ────────────────────────────────────────────────
up: ## Start postgres + PROFILES (default: pipeline). PROFILES= for postgres only
	$(COMPOSE) $(foreach p,$(PROFILES),--profile $(p)) up -d --build
down: ## Stop every service, whatever profile started it
	$(COMPOSE) $(ALL) down
clean: ## Stop everything and delete volumes (destructive)
	$(COMPOSE) $(ALL) down -v
logs: ## Tail logs of running services
	$(COMPOSE) $(ALL) logs -f --tail=100
init: ## Create the OLTP schema, lake directories and (if running) Kafka topics
	bash infra/init/bootstrap.sh

# ── pipeline ──────────────────────────────────────────────────────
ingest:    ## Download public corpora into bronze
	$(DAG) ingest_corpus
curate:    ## Normalize, scrub PII, deduplicate → silver
	$(DAG) curate_spark
generate:  ## Generate synthetic dialogues by code from their labels
	$(DAG) generate_combinatorial
augment:   ## Inject calibrated ASR noise (rule-based)
	$(DAG) augment_asr_noise
gold:      ## Build, gate and version the gold dataset (DVC), then run make tag-gold
	$(DAG) build_gold_dataset
tag-gold:  ## On the host: commit dvc.lock and create the gold.tag git tag
	$(PYTHON) -m src.pipeline.datasets.tag_release tag
train:     ## Fine-tune the generative model with QLoRA (tasks A, C; B as the H1 arm)
	$(DAG) train_qlora
train-laya: ## Fine-tune the decision model (tasks B, D)
	$(DAG) train_laya
calibrate: ## Re-fit Laya temperatures: make calibrate RUN_ID=<train_laya run>
	$(DAG) calibrate_laya --conf '{"run_id":"$(RUN_ID)"}'
analytics: ## Simulated telemetry -> Spark -> warehouse KPIs (DAG telemetry_analytics)
	$(DAG) telemetry_analytics
baselines: ## Rule-based and majority baselines on the gold test split
	$(DAG) evaluate_baselines
laya-zero-shot: ## Laya before fine-tuning on the gold test split (on the Mac, CPU)
	MLFLOW_TRACKING_URI=http://localhost:5001 $(PYTHON) -m src.pipeline.evaluate.laya_zero_shot
eval:      ## Evaluate against the frozen eval set
	$(DAG) evaluate_model

# ── agent ─────────────────────────────────────────────────────────
serve:     ## Serve Qwen + adapter with vLLM (OpenAI-compatible)
	$(PYTHON) -m src.serving.qwen
serve-laya: ## Serve Laya-multilingual, preloaded and calibrated (models/laya*)
	$(PYTHON) -m src.serving.laya
demo:      ## Voice demo in the browser, Pipecat over WebRTC: http://localhost:7860/client
	$(PYTHON) -m src.agent.demo.server
simulate:  ## Replay simulated calls to Kafka: make simulate ARGS="--calls 50 --speed 10"
	$(PYTHON) -m src.agent.telemetry.simulate $(ARGS)
demo-text: ## The same call typed in a terminal (no audio, no ElevenLabs)
	$(PYTHON) -m src.agent.demo.text

# ── exploration ───────────────────────────────────────────────────
notebook:  ## Jupyter Lab on notebooks/ (exploration only, never imported)
	$(PYTHON) -m jupyter lab notebooks/

# ── quality ───────────────────────────────────────────────────────
test:      ## Unit tests (no infrastructure required)
	$(PYTHON) -m pytest tests/unit -q
test-all:  ## All tests, infrastructure required
	$(PYTHON) -m pytest tests -q
lint:      ## Lint and type-check
	$(PYTHON) -m ruff check src dags tests && $(PYTHON) -m mypy src
fmt:       ## Auto-format
	$(PYTHON) -m ruff format src dags tests && $(PYTHON) -m ruff check --fix src dags tests

.PHONY: help up down clean logs init ingest curate generate augment gold tag-gold analytics baselines laya-zero-shot train train-laya calibrate eval serve serve-laya demo demo-text simulate notebook test test-all lint fmt
