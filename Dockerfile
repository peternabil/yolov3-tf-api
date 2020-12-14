FROM ubuntu:latest
MAINTAINER  PeterNabil "peter@nabil.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /yolovs_tf
WORKDIR /yolovs_tf
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
