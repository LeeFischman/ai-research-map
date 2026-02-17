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
    
    ui_blob = """
    <style>
        #info-tab { 
            position: fixed; top: 50%; left: -300px; width: 300px; height: 300px;
            transform: translateY(-50%); background: #111827; border: 1px solid #374151;
            border-left: none; color: #f3f4f6; transition: left 0.3s ease-in-out; 
            z-index: 99999; padding: 20px; border-radius: 0 12px 12px 0; 
            font-family: system-ui, -apple-system, sans-serif; box-shadow: 5px 0 15px rgba(0,0,0,0.5);
        }
        #info-tab.open { left: 0 !important; }
        #info-toggle { 
            position: absolute; right: -40px; top: 0; width: 40px; height: 60px; 
            background: #2563eb; color: white; cursor: pointer; display: flex; 
            align-items: center; justify-content: center; border-radius: 0 8px 8px 0;
            font-size: 20px; box-shadow: 2px 0 5px rgba(0,0,0,0.2);
        }
    </style>
    <div id="info-tab">
        <div id="info-toggle" onclick="document.getElementById('info-tab').classList.toggle('open')">⚙️</div>
        <h2 style="margin-top:0; color:#60a5fa;">The AI Research Map</h2>
        <p style="font-size:0.9rem;">Interactive 5-day view of cs.AI research.</p>
        <p style="font-size:0.8rem; color:#9ca3af;">💡 Tip: Color by <b>'Reputation'</b> in the menu.</p>
        <hr style="border:0; border-top:1px solid #374151; margin:15px 0;">
        <p style="font-size:0.85rem;">Created by <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank" style="color:#3b82f6;">Lee Fischman</a></p>
        <p style="font-size:0.75rem;"><a href="https://www.amazon.com/dp/B0GMVH6P2W" target="_blank" style="color:#3b82f6;">Check out my books on Amazon</a></p>
    </div>
    """
    with open(index_file, "r") as f: content = f.read()
    if "</body>" in content:
        with open(index_file, "w") as f: f.write(content.replace("</body>", ui_blob + "</body>"))

# --- 3. EXECUTION ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    
    client = arxiv.Client(page_size=100, delay_seconds=5)
    search = arxiv.Search(
        query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", 
        max_results=250
    )
    
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
        
        # NOTE: embedding-atlas often looks for a 'label' or 'name' column by default.
        # We rename 'title' to 'label' here to satisfy it automatically.
        df['label'] = df['title']
        
        df.to_parquet(DB_PATH, index=False)
        
        print("🧠 Building Map...")
        # REMOVED --labels to let it auto-detect the 'label' column in the parquet
        subprocess.run([
            "embedding-atlas", DB_PATH, 
            "--text", "text", 
            "--model", "allenai/specter2_base", 
            "--export-application", "site.zip"
        ], check=True)
        
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        inject_custom_ui("docs")
        print("✨ Sync Complete!")
    else:
        print("No papers found.")
