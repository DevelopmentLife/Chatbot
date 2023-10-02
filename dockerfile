# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
