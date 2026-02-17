import arxiv
import pandas as pd
import subprocess
import os
import gc
import re
import time
from datetime import datetime, timedelta, timezone
from transformers import pipeline

DB_PATH = "database.parquet"
os.makedirs("docs", exist_ok=True)

# --- 1. REPUTATION LOGIC ---
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

def calculate_reputation(row):
    score = 0
    full_text = f"{row['title']} {row['text']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 3
    authors = row.get('authors', [])
    num_authors = len(authors) if isinstance(authors, list) else 0
    if num_authors >= 8: score += 2
    elif num_authors >= 4: score += 1
    if any(k in full_text for k in ['github.com', 'huggingface.co', 'open-source']): score += 2
    if any(k in full_text for k in ['benchmark', 'sota', 'ablation']): score += 1
    return "Reputation Enhanced" if score >= 4 else "Reputation Std."

# --- 2. TOOLS ---
def generate_tldrs_local(df):
    if df.empty: return []
    summarizer = pipeline("text-generation", model="MBZUAI/LaMini-Flan-T5-248M", device=-1)
    tldrs = []
    for i, row in df.iterrows():
        try:
            res = summarizer(f"Summarize: {row['title']}", max_new_tokens=30, do_sample=False)
            tldrs.append(res[0]['generated_text'].strip())
        except: tldrs.append("Summary unavailable.")
    del summarizer
    gc.collect()
    return tldrs

def inject_custom_ui(docs_path):
    index_file = os.path.join(docs_path, "index.html")
    if not os.path.exists(index_file): return
    custom_style = "<style>#info-tab { position: fixed; top: 50%; left: -320px; width: 320px; transform: translateY(-50%); background: rgba(15, 23, 42, 0.95); border-right: 3px solid #3b82f6; color: #f8fafc; transition: all 0.4s; z-index: 9999; padding: 24px; border-radius: 0 12px 12px 0; font-family: sans-serif; } #info-tab.open { left: 0; } #info-toggle { position: absolute; right: -48px; top: 50%; width: 48px; height: 48px; background: #1e40af; cursor: pointer; display: flex; align-items: center; justify-content: center; border-radius: 0 12px 12px 0; }</style>"
    tab_html = f"<div id='info-tab'><div id='info-toggle'>⚙️</div><h2>The AI Research Map</h2><p>Color by <b>'Reputation'</b> for lab-weighted scoring.</p><hr><p>By <a href='https://www.linkedin.com/in/lee-fischman/' style='color:#60a5fa'>Lee Fischman</a></p></div><script>document.getElementById('info-toggle').onclick=lambda:document.getElementById('info-tab').classList.toggle('open');</script>".replace("lambda:", "function(){") + "}"
    with open(index_file, "r") as f: content = f.read()
    with open(index_file, "w") as f: f.write(content.replace("</head>", custom_style + "</head>").replace("</body>", tab_html + "</body>"))

# --- 3. EXECUTION ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=5)).strftime('%Y-%m-%d')
    
    # Fresh Fetch
    client = arxiv.Client(page_size=100, delay_seconds=5)
    search = arxiv.Search(query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", max_results=250)
    
    results = list(client.results(search))
    if results:
        df = pd.DataFrame([{
            "id": r.entry_id.split('/')[-1],
            "title": r.title,
            "text": f"{r.title}. {r.summary}",
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d"),
            "authors": [a.name for a in r.authors]
        } for r in results])
        
        df = df.drop_duplicates(subset='id').reset_index(drop=True)
        df['tldr'] = generate_tldrs_local(df)
        df['Reputation'] = df.apply(calculate_reputation, axis=1)
        df['label'] = df['title']
        
        df.to_parquet(DB_PATH, index=False)
        
        print("🧠 Building Map...")
        subprocess.run(["embedding-atlas", DB_PATH, "--text", "text", "--model", "allenai/specter2_base", "--export-application", "site.zip"], check=True)
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        inject_custom_ui("docs")
        print("✨ Fresh Start Complete!")
    else:
        print("No papers found in the window.")
