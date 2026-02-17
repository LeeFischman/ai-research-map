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
os.makedirs("docs", exist_ok=True)

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

# --- 2. REPUTATION LOGIC ---
def calculate_reputation(row):
    score = 0
    full_text = f"{row['title']} {row['text']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 3
    num_authors = len(row.get('authors', []))
    if num_authors >= 8: score += 2
    elif num_authors >= 4: score += 1
    if any(k in full_text for k in ['github.com', 'huggingface.co', 'open-source', 'code available']):
        score += 2
    rigor_keys = ['benchmark', 'sota', 'outperforms', 'state-of-the-art', 'comprehensive', 'ablation']
    if any(k in full_text for k in rigor_keys): score += 1
    
    # Updated labels
    return "Reputation Enhanced" if score >= 4 else "Reputation Std."

# --- 3. UTILITIES ---
def fetch_results_with_retry(client, search, max_retries=5):
    for i in range(max_retries):
        try: return list(client.results(search))
        except Exception as e:
            if "429" in str(e): time.sleep((i + 1) * 30)
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

# --- 4. UI INJECTION ---
def inject_custom_ui(docs_path):
    index_file = os.path.join(docs_path, "index.html")
    if not os.path.exists(index_file): return
    custom_style = """
    <style>
        #info-tab {
            position: fixed; top: 50%; left: -320px; width: 320px; transform: translateY(-50%);
            background: rgba(15, 23, 42, 0.95); border-right: 3px solid #3b82f6;
            backdrop-filter: blur(10px); color: #f8fafc;
            box-shadow: 10px 0 30px rgba(0,0,0,0.5); transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 9999; padding: 24px; border-radius: 0 12px 12px 0; font-family: 'Inter', system-ui, sans-serif;
        }
        #info-tab.open { left: 0; }
        #info-toggle {
            position: absolute; right: -48px; top: 50%; transform: translateY(-50%);
            width: 48px; height: 48px; background: #1e40af; color: white; display: flex; 
            align-items: center; justify-content: center; cursor: pointer; border-radius: 0 12px 12px 0;
            font-size: 20px; box-shadow: 4px 0 10px rgba(0,0,0,0.2);
        }
        #info-tab h2 { margin: 0 0 4px 0; font-size: 18px; color: #60a5fa; font-weight: 700; }
        #info-tab .author { font-size: 14px; margin-bottom: 20px; color: #94a3b8; }
        #info-tab a { color: #3b82f6; text-decoration: none; font-weight: 500; }
        #info-tab hr { border: 0; border-top: 1px solid #334155; margin: 16px 0; }
        #info-tab p { font-size: 13px; line-height: 1.6; color: #cbd5e1; margin-bottom: 12px; }
        .tip { background: rgba(59, 130, 246, 0.1); border-left: 2px solid #3b82f6; padding: 10px; font-style: italic; border-radius: 0 4px 4px 0; }
    </style>
    """
    tab_html = """
    <div id="info-tab">
        <div id="info-toggle">⚙️</div>
        <h2>The AI Research Map</h2>
        <div class="author">by <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank">Lee Fischman</a></div>
        <p>A 5-day view of <b>cs.AI</b> research clusters from arXiv.</p>
        <p class="tip">💡 Color by <b>'Reputation'</b> to highlight papers from major labs, high author counts, and code releases.</p>
        <hr>
        <p>📚 <b>Books:</b> <a href="https://www.amazon.com/dp/B0GMVH6P2W" target="_blank">Check out my books on Amazon</a></p>
        <p>🛠️ <b>Technology:</b> <a href="https://apple.github.io/embedding-atlas/" target="_blank">Embedding Atlas</a></p>
        <hr>
        <div style="font-size: 11px; color: #94a3b8;">Last Sync: """ + datetime.now(timezone.utc).strftime('%Y-%m-%d') + """ UTC</div>
    </div>
    <script>
        document.getElementById('info-toggle').onclick = function() {
            document.getElementById('info-tab').classList.toggle('open');
        };
    </script>
    """
    with open(index_file, "r") as f: content = f.read()
    if "</head>" in content: content = content.replace("</head>", custom_style + "</head>")
    if "</body>" in content: content = content.replace("</body>", tab_html + "</body>")
    with open(index_file, "w") as f: f.write(content)

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    cutoff_date = (now - timedelta(days=5)).strftime('%Y-%m-%d')
    
    if os.path.exists(DB_PATH):
        db_df = pd.read_parquet(DB_PATH)
        
        # --- MIGRATION BLOCK: Fix existing data ---
        if 'Paper_Priority' in db_df.columns:
            print("🔄 Migrating 'Paper_Priority' to 'Reputation'...")
            db_df = db_df.rename(columns={'Paper_Priority': 'Reputation'})
        
        if 'Reputation' in db_df.columns:
            db_df['Reputation'] = db_df['Reputation'].replace({
                'Standard': 'Reputation Std.',
                'High Priority': 'Reputation Enhanced'
            })
        
        db_df = db_df[db_df['date'] >= cutoff_date]
    else:
        db_df = pd.DataFrame()

    client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=10)
    start_time = now - timedelta(days=5)
    date_query = f"submittedDate:[{start_time.strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]"
    search = arxiv.Search(query=f"cat:cs.AI AND {date_query}", max_results=250)
    results = fetch_results_with_retry(client, search)
    
    if results or not db_df.empty:
        if results:
            all_fetched = pd.DataFrame([{
                "id": r.entry_id.split('/')[-1],
                "title": r.title,
                "text": f"{r.title}. {r.summary}",
                "url": r.pdf_url,
                "date": r.published.strftime("%Y-%m-%d"),
                "authors": [a.name for a in r.authors]
            } for r in results])

            if not db_df.empty:
                new_data = all_fetched[~all_fetched['id'].isin(db_df['id'])].copy()
            else:
                new_data = all_fetched.copy()

            if not new_data.empty:
                new_data['tldr'] = generate_tldrs_local(new_data)
                new_data['Reputation'] = new_data.apply(calculate_reputation, axis=1)
                combined_df = pd.concat([db_df, new_data], ignore_index=True)
            else:
                combined_df = db_df
        else:
            combined_df = db_df

        combined_df = combined_df.drop_duplicates(subset=['id'], keep='last')
        combined_df = combined_df[combined_df['date'] >= cutoff_date]
        combined_df['label'] = combined_df['title']
        combined_df.to_parquet(DB_PATH)
        
        print("🧠 Building Atlas Map...")
        subprocess.run(["embedding-atlas", DB_PATH, "--text", "text", "--model", "allenai/specter2_base", "--export-application", "site.zip"], check=True)
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        inject_custom_ui("docs")
    else:
        print("⚠️ No data.")
        with open("docs/index.html", "w") as f:
            f.write("<html><body>No papers found.</body></html>")

    print("✨ Sync Complete!")
