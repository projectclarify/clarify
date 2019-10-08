FROM gcr.io/clarify/common-base:0.2.6

ENV APP_HOME /tmp
WORKDIR $APP_HOME
COPY . .

RUN pip install -e .[tensorflow,tests]
