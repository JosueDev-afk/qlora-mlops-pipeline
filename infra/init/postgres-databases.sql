-- Runs once, on the first start of an empty volume. The operational schema
-- (schema.sql) goes into POSTGRES_DB via `make init`; Airflow keeps its
-- metadata in a database of its own so the two never share tables.
CREATE DATABASE airflow;
