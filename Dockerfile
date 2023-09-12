FROM docker-release-local.docker.awstrp.net/com/trp/ea/unity-base-docker/python/python:3.10-18


WORKDIR /app
COPY requirements.txt /app
RUN mkdir /usr/local/tmp
ENV TMPDIR=/usr/local/tmp
RUN pip install -r /app/requirements.txt

EXPOSE 31000

COPY . /app

CMD uvicorn src.main:app --port 31000 --host 0.0.0.0
