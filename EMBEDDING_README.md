# Embedding Adapter System

The job search system supports multiple embedding providers through a flexible adapter pattern. You can choose between local PyTorch embeddings or OpenAI-compatible endpoints.

## Supported Providers

### 1. Local Embeddings (Default)
- **Provider**: `local`
- **Model**: sentence-transformers
- **Default Model**: `all-MiniLM-L6-v2`
- **Dimension**: 384
- **Cost**: Free
- **Speed**: Fast
- **Privacy**: All processing is done locally

### 2. OpenAI-Compatible Embeddings
- **Provider**: `openai`
- **Implementation**: Uses the official OpenAI Python package
- **Models**: 
  - `text-embedding-3-small` (1536 dimensions)
  - `text-embedding-3-large` (3072 dimensions)
  - `text-embedding-ada-002` (1536 dimensions)
- **Cost**: Based on OpenAI pricing
- **Speed**: Fast (API-based)
- **Privacy**: Data sent to API endpoint
- **Features**: 
  - Full support for custom base URLs
  - Support for custom dimensions (text-embedding-3 models)
  - Compatible with any OpenAI-compatible API

## Configuration

### Using Local Embeddings (Default)

No additional configuration needed. The system will use local embeddings by default.

```bash
# .env file
EMBEDDING_PROVIDER = "local"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### Using OpenAI Embeddings

```bash
# .env file
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_API_KEY = "your_OPENAI_EMBEDDING_API_KEY"
```

### Using Custom OpenAI-Compatible Endpoint

You can use any OpenAI-compatible endpoint, including local LLM servers:

```bash
# .env file
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_API_KEY = "your_api_key"
OPENAI_EMBEDDING_BASE_URL = "http://localhost:8000/v1"  # Your custom endpoint
```

**Important**: When using custom OpenAI-compatible endpoints with non-OpenAI models, you must specify the embedding dimension:

```bash
# .env file
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "nomic-embed-text"  # Custom model name
OPENAI_EMBEDDING_API_KEY = "your_api_key"
OPENAI_EMBEDDING_BASE_URL = "http://localhost:11434/v1"
OPENAI_EMBEDDING_DIMENSION = 768  # Required for non-OpenAI models
```

The `OPENAI_EMBEDDING_DIMENSION` parameter is required when:
- Using custom models not in the OpenAI model list
- Using local LLM servers (Ollama, vLLM, etc.)
- Using third-party OpenAI-compatible APIs

## Usage Examples

### Programmatic Usage

```python
from embedding_adapters import get_embedding_adapter

# Use local embeddings (default)
adapter = get_embedding_adapter()
embeddings = adapter.embed(["Hello world", "Job description"])
print(f"Dimension: {adapter.get_dimension()}")

# Use OpenAI embeddings
adapter = get_embedding_adapter(
    provider="openai",
    model_name="text-embedding-3-small",
    api_key="your_api_key"
)
embeddings = adapter.embed(["Hello world"])
```

### Using with Job Processor

The job processor automatically uses the configured embedding provider:

```bash
# Set your provider in .env
export EMBEDDING_PROVIDER="openai"
export OPENAI_EMBEDDING_API_KEY="your_key"

# Run the processor
python job_processor.py
```

### Using with Web Interface

The web interface's semantic search automatically uses the configured embedding provider:

```bash
# Set your provider in .env
export EMBEDDING_PROVIDER="openai"
export OPENAI_EMBEDDING_API_KEY="your_key"

# Start the web interface
python web_interface.py
```

## Model Comparison

### Local Models (sentence-transformers)

| Model | Dimension | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | General purpose, fast |
| all-mpnet-base-v2 | 768 | Fast | Very Good | Better quality |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | Fast | Good | Multilingual |

### OpenAI Models

| Model | Dimension | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| text-embedding-3-small | 1536 | Fast | Excellent | General purpose, cost-effective |
| text-embedding-3-large | 3072 | Fast | Excellent | Best quality |
| text-embedding-ada-002 | 1536 | Fast | Good | Legacy support |

## Switching Between Providers

### From Local to OpenAI

1. Update your `.env` file:
```bash
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_API_KEY = "your_api_key"
```

2. Re-embed your jobs:
```bash
# Delete existing ChromaDB collection
rm -rf ./chroma_db

# Re-process jobs
python job_processor.py
```

### From OpenAI to Local

1. Update your `.env` file:
```bash
EMBEDDING_PROVIDER = "local"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

2. Re-embed your jobs:
```bash
# Delete existing ChromaDB collection
rm -rf ./chroma_db

# Re-process jobs
python job_processor.py
```

## Performance Considerations

### Local Embeddings
- **Pros**: Free, fast, no API calls, privacy
- **Cons**: Lower dimension, may need GPU for large batches
- **Best for**: Small to medium datasets, privacy-sensitive data

### OpenAI Embeddings
- **Pros**: Higher quality, higher dimension, no local compute needed
- **Cons**: Cost per API call, requires internet, data sent to API
- **Best for**: Large datasets, best quality needed, no local compute constraints

## Troubleshooting

### Local Embeddings Not Working

**Error**: `ImportError: sentence-transformers is required`

**Solution**: Install sentence-transformers:
```bash
pip install sentence-transformers
```

### OpenAI Embeddings Not Working

**Error**: `ValueError: OpenAI API key is required`

**Solution**: Set your API key in `.env`:
```bash
OPENAI_EMBEDDING_API_KEY = "your_api_key"
```

**Error**: `AuthenticationError`

**Solution**: Verify your API key is correct and has access to the embeddings API.

### Custom Endpoint Not Working

**Error**: Connection refused or timeout

**Solution**: 
1. Verify your custom endpoint is running
2. Check the `OPENAI_EMBEDDING_BASE_URL` is correct
3. Ensure the endpoint is OpenAI-compatible

### Embedding Dimension Mismatch

**Error**: ChromaDB dimension mismatch when switching models

**Solution**: Delete and recreate the ChromaDB collection:
```bash
rm -rf ./chroma_db
python job_processor.py
```

## Advanced Usage

### Custom Local Model

```python
from embedding_adapters import LocalEmbeddingAdapter

# Use a different local model
adapter = LocalEmbeddingAdapter(model_name="all-mpnet-base-v2")
embeddings = adapter.embed(["Your text"])
```

### Batch Embedding

```python
from embedding_adapters import get_embedding_adapter

adapter = get_embedding_adapter()

# Embed multiple texts at once
texts = ["Job 1 description", "Job 2 description", "Job 3 description"]
embeddings = adapter.embed(texts)

# embeddings is a list of lists
for i, embedding in enumerate(embeddings):
    print(f"Text {i}: dimension {len(embedding)}")
```

### Using with Custom OpenAI-Compatible Server

You can use local LLM servers like Ollama, vLLM, or any OpenAI-compatible API:

```bash
# Example with Ollama
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "nomic-embed-text"
OPENAI_EMBEDDING_API_KEY = "ollama"  # Ollama doesn't require a real key
OPENAI_EMBEDDING_BASE_URL = "http://localhost:11434/v1"
OPENAI_EMBEDDING_DIMENSION = 768  # Required: specify the model's embedding dimension
```

**Common embedding dimensions for popular models**:
- `nomic-embed-text`: 768
- `mxbai-embed-large`: 1024
- `all-MiniLM-L6-v2`: 384 (if using via OpenAI-compatible server)
- `all-mpnet-base-v2`: 768 (if using via OpenAI-compatible server)

Check your model's documentation for the correct embedding dimension.

## Best Practices

1. **Start with local embeddings** for testing and small datasets
2. **Use OpenAI embeddings** for production with large datasets
3. **Re-embed when switching models** to avoid dimension mismatches
4. **Monitor API costs** when using OpenAI embeddings
5. **Consider privacy** - local embeddings keep data on your machine
6. **Test quality** - compare semantic search results between models

## License

This embedding adapter system is part of the brave-job-search project. See LICENSE.md for details.