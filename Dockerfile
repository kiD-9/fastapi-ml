FROM python:3.10

WORKDIR /workdir

COPY requirements.txt setup.py /workdir/

COPY app/ /workdir/app/
COPY ml/ /workdir/ml/

RUN pip install -U -e requirements.txt

EXPOSE 8080

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
