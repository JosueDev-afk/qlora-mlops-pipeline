# Tracking server built from the same mlflow requirement the pipeline uses, so
# server and clients stay on compatible versions. Multi-arch via python:slim.
FROM python:3.11-slim
COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "import tomllib; p = tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']; \
print(next(d for d in p['optional-dependencies']['pipeline'] if d.startswith('mlflow')))" > /tmp/requirements.txt \
 && pip install --no-cache-dir -r /tmp/requirements.txt
EXPOSE 5000
