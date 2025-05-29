
# Similarity API

## Overview
This is a FastAPI-based REST API that calculates similarity between a query string and a list of documents. It uses the AIPipe service to generate embeddings for the query and documents, then computes cosine similarity (without external libraries) to rank documents by relevance to the query.

---

## Features
- Accepts a POST request with a query and a list of documents.
- Calls AIPipe's embedding API to get vector representations.
- Computes cosine similarity between query and document vectors.
- Returns the top 3 most similar documents.
- Uses async calls for efficiency.
- CORS enabled for all origins.
- No dependency on sklearn for cosine similarity.

---

## Installation

1. Clone the repo or copy files.
2. Make sure you have Python 3.8+ installed.
3. Install dependencies:
   ```
   pip install fastapi uvicorn httpx numpy
   ```
4. Set the environment variable for the AIPIPE token:
   - On Linux/Mac:
     ```
     export AIPIPE_TOKEN="your_token_here"
     ```
   - On Windows (PowerShell):
     ```
     setx AIPIPE_TOKEN "your_token_here"
     ```
5. Run the API:
   ```
   uvicorn similarity_api:app --reload
   ```

---

## API Usage

**Endpoint:** `POST /similarity`

**Request Body:**
```json
{
  "query": "your query string",
  "docs": ["document 1", "document 2", "document 3", "..."]
}
```

**Response:**
```json
{
  "matches": [
    "most similar doc",
    "second most similar doc",
    "third most similar doc"
  ]
}
```

---

## How it Works

1. Receives a query and a list of documents.
2. Uses AIPipe embedding endpoint asynchronously to get embeddings for all texts.
3. Computes cosine similarity for each doc vector vs. the query vector.
4. Sorts documents by similarity descending.
5. Returns top 3 most similar documents in JSON response.

---

## Deployment

You can deploy this API on platforms like **Vercel** or **Heroku**.

### Example steps for deployment on Vercel:

1. Sign up on [vercel.com](https://vercel.com).
2. Install Vercel CLI:
   ```
   npm i -g vercel
   ```
3. Initialize your project:
   ```
   vercel init
   ```
4. Deploy:
   ```
   vercel --prod
   ```
5. Set environment variables on Vercel dashboard (AIPIPE_TOKEN).
6. Your API will be live at the provided URL.

---

## Notes

- Make sure your AIPipe token has access to embedding API.
- API supports only POST requests to `/similarity`.
- No external sklearn dependency; cosine similarity is computed manually.
- Designed for easy integration with front-end apps.

---

## License

MIT License

---

## Author

Created by [Your Name]
