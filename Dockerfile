FROM python:3.10-18


WORKDIR /app
RUN adduser --system --uid 9999 appuser
RUN chown -R appuser . $WORKDIR

COPY requirements.txt /app
RUN mkdir /usr/local/tmp
ENV TMPDIR=/usr/local/tmp
RUN pip install -r /app/requirements.txt

USER appuser

EXPOSE 31000

COPY . /app

CMD uvicorn src.main:app --port 31000 --host 0.0.0.0
