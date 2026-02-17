import arxiv
import pandas as pd
import subprocess
import os
import gc
import torch
import re
import time
from datetime import datetime, timedelta, timezone
from transformers import pipeline

# --- 1. CONFIGURATION ---
INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([
    "MIT", "Stanford", "CMU", "Carnegie Mellon", "UC Berkeley", "Harvard", "Princeton", 
    "Cornell", "UWashington", "UMich", "Georgia Tech", "UT Austin", "UIUC", "NYU", 
    "UCLA", "Columbia", "UPenn", "USC", "UCSD", "UMass", "Johns Hopkins",
    "DeepMind", "OpenAI", "Anthropic", "FAIR", "Meta AI", "Microsoft Research", 
    "IBM Research", "Amazon AI", "AWS AI", "NVIDIA", "Adobe Research", "Apple Research",
    "Vector Institute", "MILA", "UBC", "McGill", "AMII", "Oxford", "Cambridge", "UCL", 
    "Imperial College", "Edinburgh", "Alan Turing", "ETH Zurich", "EPFL", "Max Planck", 
    "TUM", "Amsterdam", "KU Leuven", "INRIA", "Sorbonne", "Copenhagen", "Tsinghua", 
    "Peking", "Chinese Academy of Sciences", "SJtu", "Zhejiang", "Fudan", "Nanjing", 
    "USTC", "Harbin Institute", "Xi'an Jiaotong", "Beihang", "CUHK", "HKUST", "HKU", 
    "Baidu", "Alibaba", "DAMO", "Tencent", "Huawei", "Noah's Ark", "ByteDance", 
    "SenseTime", "Megvii", "iFlytek", "JD AI", "Xiaomi", "DiDi", "Kuaishou", "PingAn", 
    "NetEase", "Meituan", "National University of Singapore", "NUS", "UTokyo", "KAIST", 
    "SNU", "RIKEN", "Naver", "Samsung", "IISc", "IIT", "Tel Aviv", "Technion", 
    "ANU", "Melbourne", "Sydney"
]) + r")\b", re.IGNORECASE)

# --- 2. RETRY LOGIC FOR ARXIV ---
def fetch_results_with_retry(client, search, max_retries=5):
    for i in range(max_retries):
        try:
            return list(client.results(search))
        except Exception as e:
            if "429" in str(e):
                wait = (i + 1) * 30
                print(f"⚠️ Rate limited (429). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Max retries exceeded for arXiv API")

# --- 3. AI SUMMARIZATION ---
def generate_tldrs_local(df):
    if df.empty: return []
    print("🤖 Loading LaMini-Flan-T5-248M...")
    # Explicitly setting config to avoid the max_length warning
    summarizer = pipeline("text-generation", model="MBZUAI/LaMini-Flan-T5-248M", device=-1)
    
    tldrs = []
    for i, row in df.iterrows():
        prompt = f"Summarize: {row['title']}. {row['text_for_embedding'][:400]}"
        try:
            # We only use max_new_tokens here
            res = summarizer(prompt, max_new_tokens=40, do_sample=False, truncation=True)
            output = res[0]['generated_text']
            if prompt in output: output = output.replace(prompt, "").strip()
            tldrs.append(output.strip())
        except:
            tldrs.append("Summary unavailable.")
        if (i + 1) % 10 == 0: print(f"✅ Processed {i+1}/{len(df)}...")
    
    del summarizer
    gc.collect()
    return tldrs

# --- 4. MAIN ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    # Using a slightly longer delay to be safe
    client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=10)
    
    yesterday = now - timedelta(days=1)
    date_query = f"submittedDate:[{yesterday.strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]"
    
    search = arxiv.Search(query=f"cat:cs.AI AND {date_query}", max_results=100, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    results = fetch_results_with_retry(client, search)
    
    if not results:
        print("📭 No new papers.")
    else:
        df = pd.DataFrame([{
            "title": r.title,
            "text_for_embedding": f"{r.title}. {r.summary}",
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d")
        } for r in results])
        
        df['tldr'] = generate_tldrs_local(df)
        df.to_parquet("papers.parquet")
        
        print("🧠 Creating Vector Map...")
        # Removed --color flag entirely as the CLI version 0.17.0 doesn't support it
        subprocess.run([
            "embedding-atlas", "papers.parquet",
            "--text", "text_for_embedding",
            "--model", "allenai/specter2_base",
            "--export-application", "site.zip"
        ], check=True)
        
        os.makedirs("docs", exist_ok=True)
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
    print("✨ Done!")
