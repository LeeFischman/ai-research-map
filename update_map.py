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

# --- 2. UI INJECTION ---
def inject_custom_ui(docs_path):
    index_file = os.path.join(docs_path, "index.html")
    if not os.path.exists(index_file): return
    
    # Using a high-visibility, fixed-position UI
    ui_blob = """
    <div id="custom-overlay" style="position:fixed; top:20px; left:20px; z-index:2147483647;">
        <button id="info-toggle" style="background:#2563eb; color:white; border:none; padding:10px 15px; border-radius:8px; cursor:pointer; font-weight:bold; box-shadow:0 4px 6px rgba(0,0,0,0.3);">
            ⚙️ Menu
        </button>
        <div id="info-tab" style="display:none; margin-top:10px; width:280px; background:#111827; border:1px solid #374151; color:white; padding:20px; border-radius:12px; box-shadow:0 10px 15px rgba(0,0,0,0.5); font-family:sans-serif;">
            <h2 style="margin:0 0 10px 0; color:#60a5fa; font-size:18px;">AI Research Map</h2>
            <p style="font-size:13px; color:#d1d5db;">Color by <b>'Reputation'</b> to see lab-weighted scoring.</p>
            <hr style="border:0; border-top:1px solid #374151; margin:15px 0;">
            <p style="font-size:13px;">By <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank" style="color:#3b82f6;">Lee Fischman</a></p>
            <p style="font-size:12px;"><a href="https://www.amazon.com/dp/B0GMVH6P2W" target="_blank" style="color:#3b82f6;">My Books on Amazon</a></p>
        </div>
    </div>
    <script>
        document.getElementById('info-toggle').onclick = function() {
            var tab = document.getElementById('info-tab');
            tab.style.display = tab.style.display === 'none' ? 'block' : 'none';
        };
    </script>
    """
    with open(index_file, "r") as f:
        content = f.read()
    
    # Injecting immediately after body tag starts
    if "<body>" in content:
        new_content = content.replace("<body>", "<body>" + ui_blob)
        with open(index_file, "w") as f:
            f.write(new_content)

# --- 3. EXECUTION ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    
    client = arxiv.Client(page_size=100)
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
            "authors": [a.name for a in r.authors],
            "label": r.title  # Re-adding explicit label
        } for r in results])
        
        df = df.drop_duplicates(subset='id').reset_index(drop=True)
        df['Reputation'] = df.apply(calculate_reputation, axis=1)
        df.to_parquet(DB_PATH, index=False)
        
        print("🧠 Building Map...")
        # Use simple flags to avoid FileNotFoundError
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
