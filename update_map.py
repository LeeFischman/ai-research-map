import arxiv
import pandas as pd
import subprocess
import os
import sys
from datetime import datetime

def fetch_arxiv_data(limit=500):
    client = arxiv.Client()
    
    # Define your search
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    data = []
    print(f"📥 Fetching {limit} papers from arXiv...")
    try:
        # THE FIX: pass 'search' into results()
        for r in client.results(search):
            data.append({
                "title": r.title,
                "text_for_embedding": f"{r.title}. {r.summary}",
                "url": r.pdf_url,
                "date": r.published.strftime("%Y-%m-%d"),
                "category": r.primary_category
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"❌ ArXiv API Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 1. Get Data
    df = fetch_arxiv_data(500)
    print(f"📊 Collected {len(df)} papers. Saving to Parquet...")
    df.to_parquet("papers.parquet")
    
    # 2. Run Apple Embedding Atlas
    print("🧠 Generating Map (SPECTER2)...")
    try:
        # We use subprocess to call the Atlas engine
        subprocess.run([
            "embedding-atlas", "papers.parquet",
            "--text", "text_for_embedding",
            "--model", "allenai/specter2_base",
            "--export-application", "site.zip"
        ], check=True)
        
        # 3. Clean up and Unzip for Web
        os.makedirs("docs", exist_ok=True)
        os.system("unzip -o site.zip -d docs/")
        
        # 4. Create NoJekyll & Status file
        with open("docs/.nojekyll", "w") as f: f.write("")
        with open("docs/status.json", "w") as f:
            f.write(f'{{"last_update": "{datetime.now().strftime("%Y-%m-%d")}"}}')
            
        print("✅ SUCCESS: Map files generated in /docs")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Atlas Engine failed: {e}")
        sys.exit(1)
