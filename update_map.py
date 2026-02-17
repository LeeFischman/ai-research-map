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

DB_PATH = "database.parquet"

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

def fetch_results_with_retry(client, search, max_retries=5):
    for i in range(max_retries):
        try:
            return list(client.results(search))
        except Exception as e:
            if "429" in str(e):
                wait = (i + 1) * 30
                time.sleep(wait)
            else: raise e
    raise Exception("Max retries exceeded")

def generate_tldrs_local(df):
    if df.empty: return []
    summarizer = pipeline("text-generation", model="MBZUAI/LaMini-Flan-T5-248M", device=-1)
    tldrs = []
    for i, row in df.iterrows():
        prompt = f"Summarize: {row['title']}"
        try:
            res = summarizer(prompt, max_new_tokens=30, do_sample=False, max_length=None)
            tldrs.append(res[0]['generated_text'].replace(prompt, "").strip())
        except: tldrs.append("Summary unavailable.")
    del summarizer
    gc.collect()
    return tldrs

def judge_significance(row):
    score = 0
    full_text = f"{row['title']} {row['text']}".lower() 
    if INSTITUTION_PATTERN.search(full_text): score += 2
    if any(k in full_text for k in ['github.com', 'huggingface.co', 'sota', 'outperforms']): score += 1
    return "High Priority" if score >= 2 else "Standard"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    cutoff_date = (now - timedelta(days=5)).strftime('%Y-%m-%d')
    
    if os.path.exists(DB_PATH):
        db_df = pd.read_parquet(DB_PATH)
        db_df = db_df[db_df['date'] >= cutoff_date]
    else:
        db_df = pd.DataFrame()

    client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=10)
    start_time = now - timedelta(days=5)
    date_query = f"submittedDate:[{start_time.strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]"
    
    search = arxiv.Search(query=f"cat:cs.AI AND {date_query}", max_results=250)
    results = fetch_results_with_retry(client, search)
    
    if results:
        all_fetched = pd.DataFrame([{
            "id": r.entry_id.split('/')[-1],
            "title": r.title,
            "text": f"{r.title}. {r.summary}", # Standard name for main content
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d")
        } for r in results])

        if not db_df.empty:
            new_data = all_fetched[~all_fetched['id'].isin(db_df['id'])].copy()
        else:
            new_data = all_fetched.copy()

        if not new_data.empty:
            new_data['tldr'] = generate_tldrs_local(new_data)
            new_data['Paper_Priority'] = new_data.apply(judge_significance, axis=1)
            combined_df = pd.concat([db_df, new_data], ignore_index=True)
        else:
            combined_df = db_df

        combined_df = combined_df.drop_duplicates(subset=['id'], keep='last')
        combined_df = combined_df[combined_df['date'] >= cutoff_date]
        
        # We use 'label' which is a reserved keyword in most mapping engines
        combined_df['label'] = combined_df['title']

        combined_df.to_parquet(DB_PATH)
        
        print("🧠 Creating Vector Map...")
        # Simplest possible command. Let the engine auto-detect 'text' and 'label'
        subprocess.run([
            "embedding-atlas", DB_PATH,
            "--text", "text",
            "--model", "allenai/specter2_base",
            "--export-application", "site.zip"
        ], check=True)
        
        os.makedirs("docs", exist_ok=True)
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
    print("✨ Process Complete!")
