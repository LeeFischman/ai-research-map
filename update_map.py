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
    summarizer = pipeline("text-generation", model="MBZUAI/LaMini-Flan-T5-248M", device=-1)
    
    tldrs = []
    for i, row in df.iterrows():
        # Keep prompt very short to avoid truncation issues
        prompt = f"Summarize: {row['title']}"
        try:
            res = summarizer(prompt, max_new_tokens=30, do_sample=False)
            output = res[0]['generated_text']
            # Basic cleanup
            tldrs.append(output.replace(prompt, "").strip())
        except:
            tldrs.append("Summary unavailable.")
        if (i + 1) % 10 == 0: print(f"✅ Processed {i+1}/{len(df)}...")
    
    del summarizer
    gc.collect()
    return tldrs

# --- 4. SCORING LOGIC ---
def judge_significance(row):
    score = 0
    full_text = f"{row['title']} {row['text_for_embedding']}".lower()
    
    # +2 for Elite Institutions
    if INSTITUTION_PATTERN.search(full_text):
        score += 2
    # +1 for code or performance keywords
    if any(k in full_text for k in ['github.com', 'huggingface.co', 'sota', 'outperforms']):
        score += 1
        
    return "High Priority" if score >= 2 else "Standard"

# --- 5. MAIN ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=10)
    
    yesterday = now - timedelta(days=1)
    date_query = f"submittedDate:[{yesterday.strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]"
    
    search = arxiv.Search(query=f"cat:cs.AI AND {date_query}", max_results=100, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    results = fetch_results_with_retry(client, search)
    
    if not results:
        print("📭 No new papers.")
    else:
        # Create initial list
        data_list = []
        for r in results:
            data_list.append({
                "title": r.title,
                "text_for_embedding": f"{r.title}. {r.summary}",
                "url": r.pdf_url,
                "date": r.published.strftime("%Y-%m-%d")
            })
        
        df = pd.DataFrame(data_list)
        
        # Add the TL;DRs
        df['tldr'] = generate_tldrs_local(df)
        
        # IMPORTANT: Add the score column so users can select it in the UI
        df['Paper_Priority'] = df.apply(judge_significance, axis=1)
        
        # Save to parquet
        df.to_parquet("papers.parquet")
        
        print("🧠 Creating Vector Map...")
        # No --color flag here; the UI handles it via the 'Paper_Priority' column
        subprocess.run([
            "embedding-atlas", "papers.parquet",
            "--text", "text_for_embedding",
            "--model", "allenai/specter2_base",
            "--export-application", "site.zip"
        ], check=True)
        
        os.makedirs("docs", exist_ok=True)
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
    print("✨ Done!")
