# Airflow plus what the pipeline zone needs: a JVM for Spark in local mode and
# the `pipeline` dependency group. Source code is mounted, not copied, so a
# change in src/ or dags/ needs no rebuild.
FROM apache/airflow:2.9.2-python3.11

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless \
 && rm -rf /var/lib/apt/lists/*
# default-java is a symlink, so the same path works on arm64 and amd64.
ENV JAVA_HOME=/usr/lib/jvm/default-java

USER airflow
# Dependencies come from pyproject.toml, so the image cannot drift from it.
# Pinning apache-airflow keeps pip from moving Airflow while resolving them.
COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "import tomllib; p = tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['pipeline']))" > /tmp/requirements.txt \
 && pip install --no-cache-dir "apache-airflow==2.9.2" -r /tmp/requirements.txt
