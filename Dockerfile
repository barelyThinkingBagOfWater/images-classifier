FROM python:3.9

ADD Dockerfile /
ADD Dockerrun.aws.json /
ADD application.py /
ADD PredictionService.py /
ADD requirements.txt /
ADD model.pth /

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "application:app"]