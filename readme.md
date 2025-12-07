# AmbedkarGPT: RAG Pipeline & Evaluation Framework

## Project Overview
This repository contains a Retrieval-Augmented Generation (RAG) system built to answer questions based on the speeches of Dr. B.R. Ambedkar. 

**Assignment 2 Update:** This project now includes a comprehensive **Evaluation Framework** that scientifically measures the system's performance. It tests different chunking strategies and calculates industry-standard NLP metrics (ROUGE, BLEU, Hit Rate) to determine the optimal configuration for the RAG pipeline.

## Key Features
* **RAG Pipeline:** Built using LangChain, ChromaDB, and Ollama (Mistral 7B).
* **Multi-Document Support:** Ingests a corpus of 6 distinct historical speeches.
* **Automated Evaluation:** A dedicated script (`evaluation.py`) that runs 25 test questions against 3 different configuration strategies.
* **Performance Metrics:**
    * **Retrieval:** Hit Rate, Mean Reciprocal Rank (MRR).
    * **Generation:** ROUGE-L, BLEU Score, Cosine Similarity.

## Prerequisites
* **Python 3.8+**
* **Ollama:** Must be installed and running locally.
* **Mistral Model:** pulled via `ollama pull mistral`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Teertha007/assingment-2.git
    cd AmbedkarGPT-Intern-Task
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the Basic Q&A Bot (Assignment 1)
To use the interactive command-line interface with the single speech demo:
```bash
python main.py
