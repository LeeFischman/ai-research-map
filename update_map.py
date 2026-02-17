import arxiv
import pandas as pd
import subprocess
import os
import gc
import torch
import re
from datetime import datetime, timedelta, timezone
from transformers import pipeline

# --- 1. CONFIGURATION: THE 95 ELITE INSTITUTIONS ---
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

# --- 2. AI SUMMARIZATION LOGIC ---
def generate_tldrs_local(df):
    if df.empty:
        return []
    
    print("🤖 Loading LaMini-Flan-T5-248M...")
    # Switched to 'text-generation' as per the available tasks in your log
    summarizer = pipeline(
        "text-generation", 
        model="MBZUAI/LaMini-Flan-T5-248M", 
        device=-1, 
        torch_dtype=torch.float32
    )
    
    tldrs = []
    print(f"✍️ Summarizing {len(df)} papers...")
    
    for i, row in df.iterrows():
        prompt = f"Summarize this research in one short sentence: {row['title']}. {row['text_for_embedding'][:500]}"
        try:
            # Added truncation=True to handle long input strings
            res = summarizer(prompt, max_length=60, min_length=10, do_sample=False, truncation=True)
            # The output for text-generation often includes the prompt; we strip it if necessary
            output = res[0]['generated_text']
            if prompt in output:
                output = output.replace(prompt, "").strip()
            tldrs.append(output)
        except Exception as e:
            tldrs.append("Summary currently unavailable.")
            
        if (i + 1) % 10 == 0:
            print(f"✅ Processed {i+1}/{len(df)} papers...")

    del summarizer
    gc.collect()
    return tldrs

# --- 3. SIGNIFICANCE SCORING LOGIC ---
def judge_significance(row):
    score = 0
    full_text = f"{row['title']} {row['text_for_embedding']}".lower()
    
    if INSTITUTION_PATTERN.search(full_text):
        score += 2
        
    if any(k in full_text for k in ['github.com', 'huggingface.co', 'sota', 'outperforms', 'breakthrough']):
        score += 1
        
    return "Significant (Red)" if score >= 2 else "Standard"

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    print(f"📅 Build Date: {now.strftime('%Y-%m-%d %H:%M')} UTC")

    client = arxiv.Client()
    yesterday = now - timedelta(days=1)
    
    date_from = yesterday.strftime('%Y%m%d%H%M')
    date_to = now.strftime('%Y%m%d%H%M')
    date_query = f"submittedDate:[{date_from} TO {date_to}]"
    
    print(f"🔍 Querying: cat:cs.AI AND {date_query}")
    search = arxiv.Search(
        query=f"cat:cs.AI AND {date_query}",
        max_results=150,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results = list(client.results(search))
    if not results:
        print("📭 No new papers found today.")
        os.makedirs("docs", exist_ok=True)
        with open("docs/index.html", "w") as f: f.write("<h1>No new papers today.</h1>")
    else:
        data = []
        for r in results:
            data.append({
                "title": r.title,
                "text_for_embedding": f"{r.title}. {r.summary}",
                "url": r.pdf_url,
                "date": r.published.strftime("%Y-%m-%d")
            })
        
        df = pd.DataFrame(data)
        df['tldr'] = generate_tldrs_local(df)
        df['status'] = df.apply(judge_significance, axis=1)
        df.to_parquet("papers.parquet")
        
        print("🧠 Creating Vector Map...")
        subprocess.run([
            "embedding-atlas", "papers.parquet",
            "--text", "text_for_embedding",
            "--color-by", "status",
            "--model", "allenai/specter2_base",
            "--export-application", "site.zip"
        ], check=True)
        
        os.makedirs("docs", exist_ok=True)
        os.system("unzip -o site.zip -d docs/")
        with open("docs/.nojekyll", "w") as f: f.write("")

    print("✨ Map update completed successfully!")
