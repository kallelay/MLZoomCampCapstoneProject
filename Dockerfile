FROM python:3.9-slim

RUN apt-get -y update

WORKDIR /app

RUN pip3 install pipenv

COPY Pipfile Pipfile.lock  ./

RUN pipenv install --deploy --ignore-pipfile

RUN pipenv install gunicorn

COPY predict.py pipeline.bin ./

COPY . ./

EXPOSE 9696

ENTRYPOINT ["pipenv", "run", "gunicorn",  "--bind=0.0.0.0:9696", "predict:app"]
