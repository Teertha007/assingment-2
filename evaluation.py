import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk

# LangChain & RAG imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# Download NLTK data needed for BLEU
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class EvaluationFramework:
    def __init__(self):
        self.corpus_path = "./corpus"
        self.dataset_path = "./test_dataset.json"
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Removed trailing space in model name
        self.llm = Ollama(model="mistral") 
        
        # Load Test Data
        with open(self.dataset_path, 'r') as f:
            self.test_data = json.load(f)['test_questions']

    def get_metrics_scorer(self):
        return rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def calculate_cosine_sim(self, text1, text2):
        # Simple embedding-based cosine similarity
        emb1 = self.embeddings.embed_query(text1)
        emb2 = self.embeddings.embed_query(text2)
        return cosine_similarity([emb1], [emb2])[0][0]

    def build_rag_system(self, chunk_size):
        print(f"--- Building RAG with Chunk Size: {chunk_size} ---")
        
        # 1. Load Documents
        loader = DirectoryLoader(self.corpus_path, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # 2. Split (Chunking Strategy)
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # 3. Vector Store
        # Using a temporary collection name to avoid contamination
        db = Chroma.from_documents(texts, self.embeddings, collection_name=f"rag_eval_{chunk_size}")
        
        # 4. Retrieval Chain
        retriever = db.as_retriever(search_kwargs={"k": 2})
        
        # Define the custom retrieval function
        def qa_chain_invoke(question):
            # Retrieve relevant documents
            retrieved_docs = retriever.invoke(question)
            # Build context from retrieved documents
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            # Create prompt
            prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {question}"
            # Generate answer
            result = self.llm.invoke(prompt)
            
            # Return in a format similar to RetrievalQA for consistency
            return {
                "result": result,
                "source_documents": retrieved_docs
            }
        
        # Return the function itself
        return qa_chain_invoke

    def evaluate_answer_quality(self, generated_ans, ground_truth):
        # 1. ROUGE-L
        scorer = self.get_metrics_scorer()
        rouge_score = scorer.score(ground_truth, generated_ans)['rougeL'].fmeasure
        
        # 2. BLEU Score
        ref_tokens = nltk.word_tokenize(ground_truth.lower())
        cand_tokens = nltk.word_tokenize(generated_ans.lower())
        
        if not cand_tokens:
            bleu_score = 0.0
        else:
            # Using weights for 1-gram and 2-gram mainly for short answers
            bleu_score = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0))
        
        # 3. Semantic Similarity (Cosine)
        cosine_score = self.calculate_cosine_sim(generated_ans, ground_truth)
        
        return rouge_score, bleu_score, cosine_score

    def run_evaluation(self):
        # [cite_start]Define strategies: Small, Medium, Large [cite: 26-29]
        chunk_strategies = [250, 550, 900] 
        results_summary = {}

        for chunk_size in chunk_strategies:
            # This variable 'qa_chain' now holds the 'qa_chain_invoke' function
            qa_chain = self.build_rag_system(chunk_size)
            
            strategy_metrics = {
                "total_questions": 0,
                "hit_rate": 0,
                "mrr": 0,
                "avg_rouge_l": [],
                "avg_bleu": [],
                "avg_cosine": []
            }

            print(f"Testing {len(self.test_data)} questions...")
            
            for item in self.test_data:
                qid = item['id']
                question = item['question']
                ground_truth = item['ground_truth']
                gold_sources = item['source_documents']
                
                # Run Inference
                try:
                    # FIX: Call 'qa_chain' (the variable), not 'qa_chain_invoke'
                    response = qa_chain(question)
                    
                    generated_ans = response['result']
                    retrieved_docs = response['source_documents']
                    
                    # --- Retrieval Metrics ---
                    # Hit Rate: Was the correct source doc found?
                    hit = False
                    rank = 0
                    for i, doc in enumerate(retrieved_docs):
                        # Extract filename from path
                        retrieved_filename = os.path.basename(doc.metadata['source'])
                        if retrieved_filename in gold_sources:
                            hit = True
                            rank = i + 1
                            break
                    
                    if hit:
                        strategy_metrics["hit_rate"] += 1
                        strategy_metrics["mrr"] += (1.0 / rank)

                    # --- Answer Quality Metrics ---
                    if item['answerable']:
                        r_score, b_score, c_score = self.evaluate_answer_quality(generated_ans, ground_truth)
                        strategy_metrics["avg_rouge_l"].append(r_score)
                        strategy_metrics["avg_bleu"].append(b_score)
                        strategy_metrics["avg_cosine"].append(c_score)

                    strategy_metrics["total_questions"] += 1
                    print(f"Q{qid} Processed. Hit: {hit}")

                except Exception as e:
                    print(f"Error on Q{qid}: {e}")

            # Finalize averages for this strategy
            total = strategy_metrics["total_questions"]
            if total > 0:
                results_summary[f"chunk_{chunk_size}"] = {
                    "Hit Rate": round(strategy_metrics["hit_rate"] / total, 3),
                    "MRR": round(strategy_metrics["mrr"] / total, 3),
                    "Avg ROUGE-L": round(np.mean(strategy_metrics["avg_rouge_l"]), 3),
                    "Avg BLEU": round(np.mean(strategy_metrics["avg_bleu"]), 3),
                    "Avg Cosine Sim": round(np.mean(strategy_metrics["avg_cosine"]), 3)
                }

        # Output Results
        print("\n--- Final Comparative Analysis ---")
        print(json.dumps(results_summary, indent=4))
        
        with open("test_results.json", "w") as f:
            json.dump(results_summary, f, indent=4)

if __name__ == "__main__":
    evaluator = EvaluationFramework()
    evaluator.run_evaluation()