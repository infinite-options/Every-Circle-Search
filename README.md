# Every Circle Backend

This backend provides semantic search functionality for businesses, wishes, and expertise using vector embeddings.

## Prerequisites

- Docker and Docker Compose must be installed and running
- Python 3.x
- MySQL database access (configured via environment variables)

## Setup

### 0. Start Python 11 Virtual environment

**Must be Python 11**

### 1. Start Docker Services

**You must run Docker first** before starting the application. The project uses Docker Compose to set up OpenSearch and OpenSearch Dashboards. Either launch Docker Desktop or:

```bash
docker-compose up -d
```

This will start:

- OpenSearch on port `9200`
- OpenSearch Dashboards on port `5601`

### 1.5. Start Qdrant (Required for QdrntTest.py)

**If you plan to use `QdrntTest.py`, you must also start Qdrant.** Qdrant is not included in the docker-compose.yml file and needs to be run separately:

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

This will start Qdrant on:

- HTTP API on port `6333`
- gRPC API on port `6334`

**Note:** Keep this Docker container running while using `QdrntTest.py`. You can run it in the background by adding `-d` flag:

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Install Python Dependencies

**For Mac users:** You must use `requirements_mac.txt` to install dependencies:

```bash
pip install -r requirements_mac.txt
```

For other platforms, you may use `requirements.txt`, but Mac users should specifically use `requirements_mac.txt` as it contains the necessary dependencies for running the application on macOS.

### 3. Environment Configuration

Create a `.env` file in the project root with your database and service configuration:

```env
RDS_HOST=your_host
RDS_PORT=3306
RDS_USER=your_user
RDS_PW=your_password
RDS_DB=your_database
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
```

## Running the Applications

### QdrntTest.py

**When to use:** Use `QdrntTest.py` when you need to use Qdrant vector database for semantic search. This application:

- Syncs data from MySQL to Qdrant collections (businesses, wishes, expertise)
- Provides search endpoints that leverage Qdrant's vector search capabilities
- Runs on port `5001`

**How to run:**

```bash
python QdrntTest.py
```

**Endpoints:**

- `GET /search_business?q=<query>&limit=<number>`
- `GET /search_wishes?q=<query>&limit=<number>`
- `GET /search_expertise?q=<query>&limit=<number>`

**Prerequisites:**

- Qdrant must be running before starting this application (see Setup section 1.5)
- If you see a "Connection refused" error, it means Qdrant is not running. Start it using the Docker command above.

### App.py

**When to use:** Use `App.py` when you want to perform semantic search directly against MySQL without Qdrant. This application:

- Queries MySQL directly
- Performs in-memory vector similarity calculations using sentence transformers
- Supports filtering by city, rating, and distance
- Runs on port `5001`

**How to run:**

```bash
python App.py
```

**Endpoints:**

- `GET /search?q=<query>&city=<city>&min_rating=<rating>&lat=<latitude>&lon=<longitude>&radius_miles=<miles>`

**Note:** Only one application can run on port 5001 at a time. Make sure to stop one before starting the other.

## Important Notes

- **Mac Users:** Always use `requirements_mac.txt` for dependency installation on macOS
- **Docker:** Docker services must be running before starting either Python application
- **Port Conflict:** Both `QdrntTest.py` and `App.py` use port 5001, so only run one at a time
- **Qdrant:** `QdrntTest.py` requires Qdrant to be running separately (not included in docker-compose.yml). If you get a "Connection refused" error, make sure Qdrant is running using the Docker command in section 1.5
