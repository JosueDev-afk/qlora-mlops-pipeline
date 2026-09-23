.DEFAULT_GOAL := help
COMPOSE := docker compose -f infra/docker-compose.yml
DAG     := $(COMPOSE) exec -T airflow-scheduler airflow dags trigger

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

# ── infrastructure ────────────────────────────────────────────────
up: ## Start the full stack
	$(COMPOSE) up -d
down: ## Stop the stack
	$(COMPOSE) down
clean: ## Stop the stack and delete volumes (destructive)
	$(COMPOSE) down -v
logs: ## Tail logs
	$(COMPOSE) logs -f --tail=100
init: ## Create DDL, MinIO buckets and Kafka topics
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
serve:     ## Serve the model with vLLM
	python -m src.agent.llm.server
demo:      ## Web voice demo (Pipecat over WebRTC)
	python -m src.agent.demo.server
demo-text: ## Fallback text demo in Streamlit
	streamlit run src/agent/demo/streamlit_app.py

# ── quality ───────────────────────────────────────────────────────
test:      ## Unit tests (no infrastructure required)
	pytest tests/unit -q
test-all:  ## All tests, infrastructure required
	pytest tests -q
lint:      ## Lint and type-check
	ruff check src dags tests && mypy src
fmt:       ## Auto-format
	ruff format src dags tests && ruff check --fix src dags tests

.PHONY: help up down clean logs init ingest curate generate augment gold train train-laya calibrate eval serve demo demo-text test test-all lint fmt
