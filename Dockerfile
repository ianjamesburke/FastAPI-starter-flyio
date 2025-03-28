# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 6969 available to the world outside this container
EXPOSE 6969

# Changed command to run main.py directly from the app directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6969", "--app-dir", "app"]