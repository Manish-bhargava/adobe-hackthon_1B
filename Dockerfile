# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory inside the container
# All commands like COPY, RUN, CMD will be relative to /app
WORKDIR /app

# Copy your Python script into the container
COPY run_analysis.py /app/

# Create the 'documents' directory inside the container for inputs
# This directory will be where your actual PDF files are mounted during execution
RUN mkdir -p /app/documents

# Install Python dependencies
# Using specific versions for reproducibility, as installed in your Conda env
# Removed the problematic 'python -m spacy_download' line
RUN pip install PyPDF2==3.0.0 \
    spacy==3.7.4 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    rank_bm25==0.2.2 \
    sentence-transformers==2.4.0 \
    pdfminer.six==20221105 \
    && python -m spacy download en_core_web_sm

# Define the default command to run when the container starts
# This will be overridden by the actual 'docker run' command from the evaluator
CMD ["python", "run_analysis.py"]