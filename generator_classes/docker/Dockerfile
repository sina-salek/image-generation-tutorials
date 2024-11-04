# Use the official Python 3.11 image from Docker Hub
FROM python:3.11-slim

# Install Poetry
RUN pip install poetry

# Set the working directory in the container
WORKDIR /app

# Copy pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock* ../

# Install dependencies using Poetry
RUN poetry install --no-root

# Copy the rest of your application code into the container
COPY . .

# Default command to run when starting the container
CMD ["python"]
