#!/usr/bin/env bash
# Idempotent bootstrap: OLTP schema, lake directories, Kafka topics.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
COMPOSE="docker compose --env-file .env -f infra/docker-compose.yml"

echo "→ postgres schema"
$COMPOSE exec -T postgres psql -U "${POSTGRES_USER:-agent}" -d "${POSTGRES_DB:-voice_agent}" \
  -v ON_ERROR_STOP=0 < infra/init/schema.sql

echo "→ lake directories"
mkdir -p data/bronze data/silver data/gold

# Kafka belongs to the `streaming` profile; skip it when that profile is down.
if $COMPOSE ps --status running --services | grep -qx kafka; then
  echo "→ kafka topics"
  for t in calls.events calls.transcripts model.inferences; do
    $COMPOSE exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 \
      --create --if-not-exists --topic "$t" --partitions 3 --replication-factor 1
  done
else
  echo "→ kafka not running (start it with: make up PROFILES=streaming); topics skipped"
fi

echo "✓ bootstrap complete"
