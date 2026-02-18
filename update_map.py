import arxiv
import pandas as pd
import subprocess
import os
import gc
import re
import time
import shutil
from datetime import datetime, timedelta, timezone

DB_PATH = "database.parquet"

# --- 1. REPUTATION LOGIC ---
INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([
    "MIT", "Stanford", "CMU", "UC Berkeley", "Harvard", "DeepMind", "OpenAI", "Anthropic", "FAIR", "Meta AI"
]) + r")\b", re.IGNORECASE)

def calculate_reputation(row):
    score = 0
    # Search the combined text for institution matches
    full_text = f"{row['label']} {row['text']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 3
    if any(k in full_text for k in ['github.com', 'huggingface.co', 'sota']): score += 2
    return "Reputation Enhanced" if score >= 4 else "Reputation Std."

# --- 2. PATIENT ARXIV FETCHER (Fixes 429/503 errors) ---
def fetch_results_with_retry(client, search, max_retries=5):
    for i in range(max_retries):
        try:
            return list(client.results(search))
        except Exception as e:
            wait = (i + 1) * 60
            print(f"⚠️ arXiv error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded")

# --- 3. UI INJECTION ---
def inject_custom_ui(docs_path):
    index_file = os.path.join(docs_path, "index.html")
    if not os.path.exists(index_file): return
    
    ui_blob = """
    <div id="lee-overlay" style="position:fixed; top:20px; left:20px; z-index:999999;">
        <button id="info-toggle" style="background:#2563eb; color:white; border:none; padding:10px 15px; border-radius:8px; cursor:pointer; font-weight:bold; box-shadow:0 4px 6px rgba(0,0,0,0.3);">⚙️ Menu</button>
        <div id="info-tab" style="display:none; margin-top:10px; width:280px; background:#111827; border:1px solid #374151; color:white; padding:20px; border-radius:12px; font-family:sans-serif; box-shadow:0 10px 15px rgba(0,0,0,0.5);">
            <h2 style="margin:0; color:#60a5fa; font-size:18px;">AI Research Map</h2>
            <p style="font-size:13px; color:#d1d5db;">Created by <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank" style="color:#3b82f6; text-decoration:none;">Lee Fischman</a></p>
            <p style="font-size:12px; color:#9ca3af;">💡 Tip: Color by <b>'Reputation'</b> in the side menu.</p>
            <hr style="border:0; border-top:1px solid #374151; margin:15px 0;">
            <p style="font-size:12px;"><a href="https://www.amazon.com/dp/B0GMVH6P2W" target="_blank" style="color:#3b82f6; text-decoration:none;">View my books on Amazon</a></p>
        </div>
    </div>
    <script>
        document.getElementById('info-toggle').onclick = function() {
            var tab = document.getElementById('info-tab');
            tab.style.display = (tab.style.display === 'none' || tab.style.display === '') ? 'block' : 'none';
        };
    </script>
    """
    with open(index_file, "r") as f:
        content = f.read()
    if "<body>" in content:
        with open(index_file, "w") as f:
            f.write(content.replace("<body>", "<body>" + ui_blob))

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Clean slate
    if os.path.exists("docs"): shutil.rmtree("docs")
    os.makedirs("docs", exist_ok=True)

    now = datetime.now(timezone.utc)
    client = arxiv.Client(page_size=100, delay_seconds=10, num_retries=10)
    search = arxiv.Search(
        query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", 
        max_results=250
    )
    
    print("📡 Fetching from arXiv...")
    results = fetch_results_with_retry(client, search)
    
    if results:
        # THE UNIVERSAL NAME STRATEGY: 'text' and 'label'
        df = pd.DataFrame([{
            "id": r.entry_id.split('/')[-1],
            "label": r.title,                  # Universal name for display labels
            "text": f"{r.title}. {r.summary}",  # Universal name for vector content
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d"),
            "authors": [a.name for a in r.authors]
        } for r in results])

        df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        df['Reputation'] = df.apply(calculate_reputation, axis=1)
        
        # Save to Parquet
        df.to_parquet(DB_PATH, index=False)
        
        print("🧠 Creating Vector Map...")
        # Simplest command: No label flags, let auto-detection find 'label'
        subprocess.run([
            "embedding-atlas", DB_PATH,
            "--text", "text",
            "--model", "allenai/specter2_base",
            "--export-application", "site.zip"
        ], check=True)
        
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        inject_custom_ui("docs")
        print("✨ Process Complete!")
    else:
        print("No papers found.")
