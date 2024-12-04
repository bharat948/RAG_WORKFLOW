# PDF Semantic Search and Embedding System

## Overview

This project provides a robust solution for processing PDF documents, creating semantic embeddings, and performing advanced document search and retrieval. The system allows users to upload PDF files, break them down into meaningful chunks, create embeddings, and perform semantic queries across the uploaded documents.

## Features

- 📄 PDF Document Processing
- 🧩 Sentence Chunking
- 🔢 Semantic Embedding Generation
- 🔍 Intelligent Query Retrieval
- 🚀 FastAPI-based Web Service

## Technology Stack

- **Backend**: Python
- **Web Framework**: FastAPI
- **Machine Learning**: 
  - Sentence Transformers
  - MPNet Encoder
- **Data Processing**: 
  - Pandas
  - NumPy
  - PyTorch
- **PDF Parsing**: PyMuPDF (fitz)

## Prerequisites

- Python 3.8+
- pip
- CUDA-compatible GPU (recommended, but not required)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-embedding-search.git
cd pdf-embedding-search
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

1. Download embedding model:
   - Place your MPNet embedding model in the `embedModel` directory
   - Ensure model supports 768-dimensional embeddings

2. Create data directory:
```bash
mkdir data
```

## Running the Application

### Start FastAPI Server
```bash
uvicorn main:app --reload
```

### API Endpoints

#### Upload PDFs
- **POST** `/upload_pdf`
  - Upload single PDF
  - Generates semantic embeddings
  - Saves embeddings to CSV

- **POST** `/upload_multiple_pdfs`
  - Upload multiple PDFs
  - Generates cumulative embeddings
  - Saves combined embeddings

#### Semantic Query
- **POST** `/query`
  - Perform semantic search across uploaded documents
  - Retrieves most relevant chunks
  - Generates context-aware responses

## Usage Example

### Upload PDF
```python
import requests

url = "http://localhost:8000/upload_pdf"
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files, data={"document_name": "my_document"})
```

### Query Document
```python
query_data = {
    "text": "What is the main topic of this document?",
    "document_name": "my_document",
    "n_results": 5
}
response = requests.post("http://localhost:8000/query", json=query_data)
```

## Configuration

- Modify `embedModel` path for custom embedding models
- Adjust chunking parameters in `ChunkCreator`
- Customize embedding generation in `EmbeddingCreator`

## Performance Optimization

- Supports batch processing
- GPU acceleration for embedding generation
- Configurable embedding chunk size

## Troubleshooting

- Ensure all dependencies are installed
- Check embedding model compatibility
- Verify document dimensions match model requirements

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

bharat patidar - bharatpatidar002@gmail.com.com

Project Link: [https://github.com/yourusername/pdf-embedding-search](https://github.com/yourusername/pdf-embedding-search)
