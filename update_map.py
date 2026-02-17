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

# --- 1. CONFIGURATION & UTILS ---
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

def judge_significance(row):
    score = 0
    full_text = f"{row['title']} {row['text']}".lower() 
    if INSTITUTION_PATTERN.search(full_text): score += 2
    if any(k in full_text for k in ['github.com', 'huggingface.co', 'sota']): score += 1
    return "High Priority" if score >= 2 else "Standard"

# --- 2. DARK MODE UI INJECTION ---
def inject_custom_ui(docs_path):
    index_file = os.path.join(docs_path, "index.html")
    if not os.path.exists(index_file): return

    custom_style = """
    <style>
        #info-tab {
            position: fixed; top: 20px; right: -320px; width: 320px;
            background: rgba(15, 23, 42, 0.95); border-left: 3px solid #3b82f6;
            backdrop-filter: blur(10px); color: #f8fafc;
            box-shadow: -10px 0 30px rgba(0,0,0,0.5); transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 9999; padding: 24px; border-radius: 12px 0 0 12px; font-family: 'Inter', system-ui, sans-serif;
        }
        #info-tab.open { right: 0; }
        #info-toggle {
            position: absolute; left: -48px; top: 0; width: 48px; height: 48px;
            background: #1e40af; color: white; display: flex; align-items: center;
            justify-content: center; cursor: pointer; border-radius: 12px 0 0 12px;
            font-size: 20px; box-shadow: -4px 0 10px rgba(0,0,0,0.2);
        }
        #info-tab h2 { margin: 0 0 4px 0; font-size: 18px; color: #60a5fa; font-weight: 700; letter-spacing: -0.025em; }
        #info-tab .author { font-size: 14px; margin-bottom: 20px; color: #94a3b8; }
        #info-tab a { color: #3b82f6; text-decoration: none; transition: color 0.2s; }
        #info-tab a:hover { color: #93c5fd; text-decoration: underline; }
        #info-tab hr { border: 0; border-top: 1px solid #334155; margin: 16px 0; }
        #info-tab p { font-size: 13px; line-height: 1.6; color: #cbd5e1; margin-bottom: 12px; }
        .badge { background: #1e293b; padding: 4px 8px; border-radius: 4px; font-size: 11px; border: 1px solid #334155; }
    </style>
    """
    
    tab_html = """
    <div id="info-tab">
        <div id="info-toggle">⚙️</div>
        <h2>The AI Research Map</h2>
        <div class="author">by <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank">Lee Fischman</a></div>
        
        <p>A rolling 5-day semantic visualization of <b>cs.AI</b> research clusters from arXiv.</p>
        
        <hr>
        
        <p>📚 <b>Latest Work:</b><br><a href="https://www.amazon.com/dp/B0GMVH6P2W" target="_blank">Check out my books on Amazon</a></p>
        
        <p>🛠️ <b>Technology:</b><br>Powered by <a href="https://apple.github.io/embedding-atlas/" target="_blank">Embedding Atlas</a> & LaMini-T5.</p>
        
        <hr>
        
        <div class="badge">Last Sync: """ + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M') + """ UTC</div>
    </div>
    <script>
        document.getElementById('info-toggle').onclick = function() {
            document.getElementById('info-tab').classList.toggle('open');
        };
    </script>
    """

    with open(index_file, "r") as f:
        content = f.read()

    if "</head>" in content:
        content = content.replace("</head>", custom_style + "</head>")
    if "</body>" in content:
        content = content.replace("</body>", tab_html + "</body>")

    with open(index_file, "w") as f:
        f.write(content)

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    cutoff_date = (now - timedelta(days=5)).strftime('%Y-%m-%d')
    
    if os.path.exists(DB_PATH):
        db_df = pd.read_parquet(DB_PATH)
        db_df = db_df[db_df['date'] >= cutoff_date]
    else: db_df = pd.DataFrame()

    client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=10)
    start_time = now - timedelta(days=5)
    date_query = f"submittedDate:[{start_time.strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]"
    
    search = arxiv.Search(query=f"cat:cs.AI AND {date_query}", max_results=250)
    results = fetch_results_with_retry(client, search)
    
    if results:
        all_fetched = pd.DataFrame([{
            "id": r.entry_id.split('/')[-1],
            "title": r.title,
            "text": f"{r.title}. {r.summary}",
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d")
        } for r in results])

        if not db_df.empty:
            new_data = all_fetched[~all_fetched['id'].isin(db_df['id'])].copy()
        else: new_data = all_fetched.copy()

        if not new_data.empty:
            new_data['tldr'] = generate_tldrs_local(new_data)
            new_data['Paper_Priority'] = new_data.apply(judge_significance, axis=1)
            combined_df = pd.concat([db_df, new_data], ignore_index=True)
        else: combined_df = db_df

        combined_df = combined_df.drop_duplicates(subset=['id'], keep='last')
        combined_df = combined_df[combined_df['date'] >= cutoff_date]
        combined_df['label'] = combined_df['title']
        combined_df.to_parquet(DB_PATH)
        
        print("🧠 Creating Vector Map...")
        subprocess.run(["embedding-atlas", DB_PATH, "--text", "text", "--model", "allenai/specter2_base", "--export-application", "site.zip"], check=True)
        
        os.makedirs("docs", exist_ok=True)
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        
        print("💉 Injecting Dark Mode UI...")
        inject_custom_ui("docs")

    print("✨ Process Complete!")
