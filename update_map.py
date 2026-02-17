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
            res = summarizer(prompt, max_new_tokens=30, do_sample=False)
            tldrs.append(res[0]['generated_text'].replace(prompt, "").strip())
        except: tldrs.append("Summary unavailable.")
    del summarizer
    gc.collect()
    return tldrs

def judge_significance(row):
    score = 0
    full_text = f"{row['title']} {row['text_for_embedding']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 2
    if any(k in full_text for k in ['github.com', 'huggingface.co', 'sota']): score += 1
    return "High Priority" if score >= 2 else "Standard"

# --- MAIN LOGIC ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    cutoff_date = (now - timedelta(days=4)).strftime('%Y-%m-%d')
    
    # 1. Load existing database
    if os.path.exists(DB_PATH):
        db_df = pd.read_parquet(DB_PATH)
        # Keep only last 4 days
        db_df = db_df[db_df['date'] >= cutoff_date]
        print(f"Loaded existing database. Rows after filtering: {len(db_df)}")
    else:
        db_df = pd.DataFrame()
        print("No database found. Creating fresh.")

    # 2. Fetch today's papers
    client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=10)
    yesterday = now - timedelta(days=1)
    date_query = f"submittedDate:[{yesterday.strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]"
    search = arxiv.Search(query=f"cat:cs.AI AND {date_query}", max_results=100)
    
    results = fetch_results_with_retry(client, search)
    
    if results:
        new_data = pd.DataFrame([{
            "id": r.entry_id.split('/')[-1], # Unique ID for deduplication
            "title": r.title,
            "text_for_embedding": f"{r.title}. {r.summary}",
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d")
        } for r in results])

        # Generate TLDR and Priority only for NEW papers to save time/compute
        new_data['tldr'] = generate_tldrs_local(new_data)
        new_data['Paper_Priority'] = new_data.apply(judge_significance, axis=1)

        # 3. Merge and Overwrite (Deduplicate by 'id')
        combined_df = pd.concat([db_df, new_data], ignore_index=True)
        # Keep the last occurrence (the newest version) of any paper ID
        combined_df = combined_df.drop_duplicates(subset=['id'], keep='last')
        
        # Final filter to ensure rolling window
        combined_df = combined_df[combined_df['date'] >= cutoff_date]
        
        # 4. Save Database
        combined_df.to_parquet(DB_PATH)
        print(f"Database saved with {len(combined_df)} total papers.")
        
        # 5. Build Map
        subprocess.run([
            "embedding-atlas", DB_PATH,
            "--text", "text_for_embedding",
            "--model", "allenai/specter2_base",
            "--export-application", "site.zip"
        ], check=True)
        
        os.makedirs("docs", exist_ok=True)
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
    else:
        print("No new papers today.")

    print("✨ Done!")
