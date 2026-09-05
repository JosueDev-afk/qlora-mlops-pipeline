#!/usr/bin/env bash
# Idempotent bootstrap: schema, MinIO buckets, Kafka topics.
set -euo pipefail
COMPOSE="docker compose -f infra/docker-compose.yml"

echo "→ postgres schema"
$COMPOSE exec -T postgres psql -U "${POSTGRES_USER:-agent}" -d "${POSTGRES_DB:-voice_agent}" \
  -v ON_ERROR_STOP=0 < infra/init/schema.sql

echo "→ minio buckets"
$COMPOSE exec -T minio sh -c '
  mc alias set local http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null
  for b in bronze silver gold mlflow dvc; do mc mb --ignore-existing "local/$b"; done'

echo "→ kafka topics"
for t in calls.events calls.transcripts model.inferences; do
  $COMPOSE exec -T kafka kafka-topics --bootstrap-server localhost:9092 \
    --create --if-not-exists --topic "$t" --partitions 3 --replication-factor 1
done

echo "✓ bootstrap complete"
