#pull the python image from dockerhub
#use a lightweight version of python to reduce the size of the image

FROM python:3.14-slim

WORKDIR /app

#copy requirements.txt 
COPY requirements.txt .
#before coppying application code
RUN pip install -r requirements.txt
#copy application code and saved model inside the app directory
COPY . .
#document the port used by the application
EXPOSE 8000
#run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#start the application server when the container is ready to serve requests
# 0.0.0.0 allows connections through the container's network interface
