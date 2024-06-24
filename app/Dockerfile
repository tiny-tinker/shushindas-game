FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install gunicorn
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app