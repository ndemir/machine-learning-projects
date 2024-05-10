# Hybrid Search Implementation

This folder contains a sample implementation combining vector search (via faiss) and keyword search (via Elasticsearch).

## How to use

# Step.1: Install Elasticsearch and run 

```
docker build -t elasticsearch . && docker run --network=host -it --rm elasticsearch
```

# Step.2: Install requirements.txt

```
pip install -r requirements.txt
```

# Step.3: Run the code

```
python main.py
```