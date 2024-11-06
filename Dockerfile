FROM python:3.11

COPY requirements.txt requirements-dev.txt setup.py /workdir/
COPY app/ /workdir/app/
COPY ml/ /workdir/ml/

WORKDIR /workdir

ADD start.sh /
RUN chmod +x /start.sh

CMD ["/start.sh"]