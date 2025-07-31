# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code from the host to the container at /app
COPY . .

# Set the PYTHONPATH to include the src directory so the innovate package can be found
ENV PYTHONPATH "${PYTHONPATH}:/app/src"

# Command to run when the container launches
CMD ["/bin/bash"]
