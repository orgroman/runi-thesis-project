# Patent Negation Analysis Workflow

This project implements a robust and scalable workflow for analyzing negation patterns in patent texts using OpenAI's batch processing API and Temporal.io for workflow orchestration.

## Architecture

The workflow uses Temporal.io to coordinate a series of activities that:
1. Authenticate with Azure Key Vault to retrieve OpenAI API key
2. Load patent data from CSV files
3. Prepare JSONL batches for OpenAI processing
4. Upload files to OpenAI and submit batch processing requests
5. Monitor batch processing status and handle completions/failures
6. Process and store results in MongoDB

The system is designed to be resilient, with automatic retries, error handling, and state persistence.

## Setup

### Prerequisites

- Python 3.8+
- MongoDB
- Temporal Server
- Azure Key Vault (for API key storage)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/runi-thesis-project.git
cd runi-thesis-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# MongoDB connection
export MONGO_URI="mongodb://localhost:27017/"
export MONGO_DB="patent_negation"

# Azure authentication (if using DefaultAzureCredential)
# export AZURE_CLIENT_ID="your-client-id"
# export AZURE_TENANT_ID="your-tenant-id"
# export AZURE_CLIENT_SECRET="your-client-secret"
```

## Usage

1. Start the Temporal worker:
```bash
python worker.py
```

2. Run the workflow:
```bash
python starter.py --csv /path/to/patents.csv --batch-size 1000
```

## Project Structure

- `workflow.py` - Main Temporal workflow definition
- `activities/` - Individual activity implementations
- `models.py` - Pydantic data models
- `mongodb.py` - MongoDB connection utilities
- `worker.py` - Temporal worker setup
- `starter.py` - Workflow execution script

## MongoDB Collections

The system uses the following MongoDB collections:
- `files` - Tracks JSONL file uploads and OpenAI file registration
- `batch_requests` - Tracks OpenAI batch processing status and results
- `negation_results` - Stores successful negation analysis results
- `batch_results_raw` - Stores raw batch processing results
- `processing_errors` - Records errors encountered during processing
- `failed_batches` - Records batch failures for troubleshooting

## Error Handling

The workflow implements comprehensive error handling:
- Activity retries with exponential backoff
- Rate limit detection and backoff
- Batch cancellation for failed requests
- Detailed error logging and MongoDB storage

## Monitoring

- Check workflow status via Temporal Web UI
- Monitor MongoDB collections for processing status
- Check log files for detailed execution information
