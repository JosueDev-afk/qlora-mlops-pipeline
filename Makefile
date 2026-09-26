.DEFAULT_GOAL := help
COMPOSE  := docker compose --env-file .env -f infra/docker-compose.yml
ALL      := --profile pipeline --profile streaming --profile bi --profile hdfs
PROFILES ?= pipeline
DAG      := $(COMPOSE) exec -T airflow-scheduler airflow dags trigger

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

# ── infrastructure ────────────────────────────────────────────────
up: ## Start postgres + PROFILES (default: pipeline). make up PROFILES="pipeline streaming"
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
gold:      ## Build, version and tag the gold dataset
	$(DAG) build_gold_dataset
train:     ## Fine-tune the generative model with QLoRA (tasks A, C; B as the H1 arm)
	$(DAG) train_qlora
train-laya: ## Fine-tune the decision model (tasks B, D)
	$(DAG) train_laya
calibrate: ## Re-fit Laya temperatures: make calibrate RUN_ID=<train_laya run>
	$(DAG) calibrate_laya --conf '{"run_id":"$(RUN_ID)"}'
eval:      ## Evaluate against the frozen eval set
	$(DAG) evaluate_model

# ── agent ─────────────────────────────────────────────────────────
serve:     ## Serve Qwen + adapter with vLLM (OpenAI-compatible)
	python -m src.serving.qwen
serve-laya: ## Serve Laya-multilingual, preloaded and calibrated (models/laya*)
	python -m src.serving.laya
demo:      ## Voice demo in the browser, Pipecat over WebRTC: http://localhost:7860/client
	python -m src.agent.demo.server
simulate:  ## Replay simulated calls to Kafka: make simulate ARGS="--calls 50 --speed 10"
	python -m src.agent.telemetry.simulate $(ARGS)
demo-text: ## The same call typed in a terminal (no audio, no ElevenLabs)
	python -m src.agent.demo.text

# ── exploration ───────────────────────────────────────────────────
notebook:  ## Jupyter Lab on notebooks/ (exploration only, never imported)
	jupyter lab notebooks/

# ── quality ───────────────────────────────────────────────────────
test:      ## Unit tests (no infrastructure required)
	pytest tests/unit -q
test-all:  ## All tests, infrastructure required
	pytest tests -q
lint:      ## Lint and type-check
	ruff check src dags tests && mypy src
fmt:       ## Auto-format
	ruff format src dags tests && ruff check --fix src dags tests

.PHONY: help up down clean logs init ingest curate generate augment gold train train-laya calibrate eval serve serve-laya demo demo-text simulate notebook test test-all lint fmt
