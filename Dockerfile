#Official python sim image
FROM python:3.9-slim-bullseye

#Install system dependencies 
RUN apt update -y && apt install awscli -y

#Set working directory 
WORKDIR /app

#Copy application files 
COPY . /app

#Install python dependencies 
RUN pip install -r requirements.txt
RUN pip install upgrade accelerate 
RUN pip uninstall -y transformers accelerate
RUN pip install transformers accelerate

#Run appliacation
CMD ["python3", "app.py"]