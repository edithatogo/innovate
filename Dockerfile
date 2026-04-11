# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set the working directory in the container
WORKDIR /app

# Copy the project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Set the PYTHONPATH to include the src directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Command to run when the container launches
CMD ["/bin/bash"]
