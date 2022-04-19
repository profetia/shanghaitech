FROM python:3.9.12-bullseye

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY src ./src

WORKDIR /usr/src/app/src

CMD [ "python", "./manage.py", "runserver", "0.0.0.0:12345" ]