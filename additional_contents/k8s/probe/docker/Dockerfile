FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ENTRYPOINT ["python"]
CMD ["app/app.py"]
