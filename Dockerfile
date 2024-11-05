FROM python:3.11

COPY requirements.txt requirements-dev.txt setup.py /workdir/
COPY app/ /workdir/app/
COPY ml/ /workdir/ml/

WORKDIR /workdir

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
CMD ["uvicorn", "app.app:app", "--host", "127.0.0.1", "--port", "80"]