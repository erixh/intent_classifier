# Intent Classifier Pipeline

A comprehensive pipeline for building intent classification datasets from web interactions using Scrapy, domain-aware BM25 filtering, and LLM-powered pseudo-labeling.


This project creates high-quality training data for intent classification by:
1. **Scraping web interactions** from various domains (Amazon, Google Docs, GitHub, etc.)
2. **Filtering actions by relevance** using domain-aware BM25 scoring
3. **Pseudo-labeling with LLMs** via Ollama integration
4. **Generating clean training datasets** for machine learning models


```
Web Scraping → Data Processing → Intent Classification → Training Data
    ↓              ↓                    ↓                  ↓
 Scrapy Spider → SQLite Database → Ollama/Mistral → JSONL Dataset
```

### Components:

- **`engenium/`** - Scrapy project for web scraping
- **`UTH_conversion/`** - Data processing pipeline  
- **`ollama_classifier.py`** - FastAPI server with Mistral integration
- **Output** - Clean intent classification training data


### Prerequisites

```bash
# Python packages
pip install scrapy fastapi uvicorn httpx aiofiles rank-bm25 requests tqdm

# Ollama with Mistral model
# Install Ollama from: https://ollama.com/download
ollama pull mistral
ollama serve
```

### 1. Scrape Web Data

```bash
cd engenium
scrapy crawl intent_scraper -o intents.jsonl
```

### 2. Process Data Pipeline

```bash
cd ../UTH_conversion
python pipeline_execution.py
```

### 3. Start LLM Classifier (Optional)

```bash
python ollama_classifier.py
```

## Pipeline 

### Step 1: Web Scraping (`engenium/`)

The Scrapy spider extracts:
- **Page URLs and titles**
- **Interactive elements** (buttons, links, forms)
- **Domain information** for context-aware processing

**Key Features:**
- Domain filtering (currently focused on Amazon)
- Rate limiting and respectful scraping
- Structured data extraction

### Step 2: Data Processing (`UTH_conversion/`)

#### A. File Conversion (`file_converter.py`)
- Converts JSONL to SQLite database
- Normalizes action data structure
- Prepares for downstream processing

#### B. BM25 Filtering (`bm25_filter.py`)  
- **Domain-aware scoring** - Different keywords for different sites:
  - **Amazon**: `["buy", "add", "cart", "purchase", "order"]`
  - **Google Docs**: `["write", "edit", "document", "format"]`
  - **GitHub**: `["commit", "push", "code", "repository"]`
- Scores actions by relevance to domain purpose
- Filters noise and focuses on meaningful interactions

#### C. Pseudo-Labeling (`self_labeling.py`)
- Integrates with Ollama/Mistral via FastAPI
- Generates natural language intent descriptions
- Fallback error handling for API failures
- Clean data validation

#### D. Training Data Export (`cleaned_data.py`)
- Exports final clean dataset as JSONL
- Format: `{"input": "domain - action", "label": "intent"}`
- Ready for machine learning model training

### Step 3: LLM Integration (`ollama_classifier.py`)

FastAPI server providing:
- **Real-time intent classification** via Mistral
- **Batch processing** capabilities
- **Health monitoring** for Ollama connectivity
- **Automatic data logging** to JSONL

## Configuration

### Scrapy Settings (`engenium/engenium/settings.py`)
```python
# Adjust scraping behavior
DOWNLOAD_DELAY = 2
CONCURRENT_REQUESTS = 1
DEPTH_LIMIT = 2
```

### Domain Keywords (`UTH_conversion/bm25_filter.py`)
```python
domain_keywords = {
    "amazon.com": ["buy", "add", "cart", "purchase"],
    "docs.google.com": ["write", "edit", "document"],
    # Add more domains as needed
}
```

### Ollama Model (`ollama_classifier.py`)
```python
# Change model if needed
OLLAMA_URL = "http://localhost:11434"
MODEL = "mistral"  # or "llama2", "codellama", etc.
```

## Example Output

### Raw Scraped Data:
```json
{
  "url": "https://amazon.com/s?k=laptop",
  "domain": "amazon.com", 
  "page_title": "Amazon.com: laptop",
  "visible_actions": ["Add to Cart", "Buy Now", "Add to Wishlist"]
}
```

### Processed Training Data:
```json
{
  "input": "amazon.com - Add to Cart",
  "output": "Add item to shopping cart for later purchase"
}
```

## Usage Examples

### Run Full Pipeline
```bash
# Complete workflow
cd engenium && scrapy crawl intent_scraper
cd ../UTH_conversion && python pipeline_execution.py
```

### Test Ollama Integration
```bash
# Start classifier API
python ollama_classifier.py

# Test classification
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "what is the user intent when they click Add to Cart on amazon.com?"}'
```

### Batch Processing
```bash
# Process multiple prompts at once
curl -X POST "http://localhost:8000/batch-label" \
     -H "Content-Type: application/json" \
     -d '["prompt1", "prompt2", "prompt3"]'
```

##  Use Cases

- **Training intent classifiers** for e-commerce platforms
- **Building chatbot training data** with real user interactions  
- **Analyzing user behavior patterns** across different domains
- **Creating domain-specific intent taxonomies**

##  Monitoring & Debugging

### Check Pipeline Health
```bash
# Ollama status
curl http://localhost:8000/health

# Database contents
sqlite3 UTH_conversion/intents.db "SELECT COUNT(*) FROM intents;"

# View generated training data
head -n 5 pseudo_labels.jsonl
```

### Common Issues

1. **Ollama not running**: Start with `ollama serve`
2. **No training data**: Check if scraping produced `intents.jsonl`
3. **API failures**: Review `self_labeling.py` error logs
4. **Empty database**: Verify file paths in `pipeline_execution.py`

##  Dependencies

```bash
# Core dependencies
pip install scrapy==2.13.1
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install httpx==0.25.2
pip install aiofiles==23.2.1
pip install rank-bm25==0.2.2
pip install requests==2.31.0
pip install tqdm==4.66.1

# Optional: For advanced processing
pip install pandas numpy scikit-learn
```

## Scaling & Extensions

### Adding New Domains
1. Update spider `allowed_domains` in `content_spider.py`
2. Add domain keywords to `bm25_filter.py`
3. Adjust intent categories in `ollama_classifier.py`

### Alternative LLMs
- Replace Ollama with OpenAI API
- Use Hugging Face transformers locally
- Integrate with cloud LLM services

### Advanced Features
- Multi-language support
- Real-time streaming classification
- Active learning for model improvement
- Custom domain adaptation

##  Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Scrapy** for powerful web scraping capabilities
- **Ollama** for local LLM inference
- **Mistral AI** for the excellent 7B model
- **FastAPI** for the robust API framework
