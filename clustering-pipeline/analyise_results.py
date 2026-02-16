import pandas as pd
import numpy as np
import os
import time
from google import genai 
from dotenv import load_dotenv

# =============================================================================
# AYARLAR
# =============================================================================
CSV_PATH = 'clustering-pipeline/results/final_results.csv'
REPORT_PATH = 'clustering-pipeline/results/cluster_analysis_report.txt'


load_dotenv()

# --- GEMINI API SETTINGS---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("API key cannot be found. Please check your .env file.")


# Gerçek AI Kullanmak için bunu False yap
DEMO_MODE = False 

FEATURES = [
    'Basic_Sciences', 'Software_Practice', 'Algorithm_Theory', 
    'Systems', 'Hardware', 'Social_Cultural', 
    'Language_Comm', 'Math_Calc'
]

# İstemciyi (Client) Hazırla
client = None
if not DEMO_MODE:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Client Init Error: {e}")

# =============================================================================
# GERÇEK AI FONKSİYONU (LİSTENİZE GÖRE GÜNCELLENDİ)
# =============================================================================
def get_ai_suggestion(cluster_id, label, strengths, weaknesses, avg_score):
    """
    Cluster verisini Gemini'ye gönderir. 
    Paylaştığın listedeki modelleri sırayla dener.
    """
    
    prompt = f"""
    You are an expert University Academic Advisor AI. 
    Analyze the following student group (Cluster {cluster_id}) based on their performance data:
    
    - Profile Label: {label}
    - Average GPA (Normalized 0-1): {avg_score:.2f}
    - Strong Areas: {strengths}
    - Weak Areas: {weaknesses}
    
    Task: Provide a specific, actionable strategic recommendation for this group (max 2 sentences).
    Do not be generic. Be specific to the subjects listed.
    """

    if DEMO_MODE:
        return "AI Simulation: Enroll in 'CS Fundamentals' bootcamp."

    if client is None:
        return "Error: Client not initialized."

    # LİSTENİZE GÖRE GÜNCELLENMİŞ MODEL LİSTESİ
    # 1. Tercih: Gemini 2.5 Flash
    # 2. Tercih: Gemini 2.0 Flash (ve varyasyonları)
    models_to_try = [
        "gemini-2.5-flash", 
        "gemini-2.0-flash", 
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash"
    ]
    
    last_error = ""

    for model_name in models_to_try:
        try:
            # İstek gönder
            response = client.models.generate_content(
                model=model_name, 
                contents=prompt
            )
            # Başarılı olursa model ismini de ekleyerek döndür
            return f"[{model_name}] {response.text.strip()}"
        except Exception as e:
            last_error = str(e)
            # Hata verirse bir sonraki modeli dene
            continue
            
    return f"All Gemini Models Failed. Last Error: {last_error}"

# =============================================================================
# ANA ANALİZ MANTIĞI
# =============================================================================
def generate_cluster_report():
    if not os.path.exists(CSV_PATH):
        if os.path.exists('final_results.csv'):
            df = pd.read_csv('final_results.csv')
        else:
            print(f"ERROR: Could not find {CSV_PATH}")
            return
    else:
        df = pd.read_csv(CSV_PATH)
    
    cluster_col = 'Cluster_DEC' if 'Cluster_DEC' in df.columns else 'Cluster'
    
    # İstatistikler
    cluster_means = df.groupby(cluster_col)[FEATURES].mean()
    cluster_counts = df[cluster_col].value_counts().sort_index()
    global_mean = df[FEATURES].mean()
    total_students = len(df)
    
    lines = []
    lines.append("==========================================================")
    lines.append("         STUDENT CLUSTER ANALYSIS REPORT (POWERED BY GEMINI)")
    lines.append("==========================================================")
    lines.append(f"Total Students: {total_students}")
    lines.append(f"Detected Clusters: {len(cluster_means)}")
    lines.append("-" * 58 + "\n")
    
    print("Generating report with Gemini suggestions...")
    print("NOTE: Waiting 13 seconds between requests to respect API Rate Limits (5 RPM)...")
    
    for c_id in cluster_means.index:
        means = cluster_means.loc[c_id]
        count = cluster_counts[c_id]
        pct = (count / total_students) * 100
        
        # 1. Güçlü/Zayıf Yönleri Belirle
        strengths = [] 
        weaknesses = [] 
        for feat in FEATURES:
            val = means[feat]
            glob_val = global_mean[feat]
            if val > 0.85 or val > (glob_val + 0.15):
                strengths.append(f"{feat}")
            elif val < 0.35 or val < (glob_val - 0.15):
                weaknesses.append(f"{feat}")
        
        # 2. Etiket Belirle
        label = "Balanced / General Group"
        avg_score = means.mean()
        
        if len(weaknesses) >= 4:
            label = "⚠️ AT-RISK / Low Engagement"
        elif avg_score > 0.90:
            label = "🏆 ELITE / High Achievers"
        elif avg_score > 0.80 and len(weaknesses) == 0:
            label = "✅ Strong & Consistent"
        elif 'Math_Calc' in str(strengths) and 'Algorithm_Theory' in str(strengths):
            label = "📐 Theoretical & Analytical Focus"
        elif 'Software_Practice' in str(strengths) and 'Systems' in str(strengths):
            label = "⚙️ Engineering & Systems Focus"
        elif 'Social_Cultural' in str(strengths) and len(strengths) <= 2:
            label = "🗣️ Socially Active / Academic Struggle"

        # 3. GEMINI TAVSİYESİ AL
        str_text = ", ".join(strengths) if strengths else "None"
        weak_text = ", ".join(weaknesses) if weaknesses else "None"
        
        ai_recommendation = get_ai_suggestion(c_id, label, str_text, weak_text, avg_score)
        
        # 4. Rapora Yaz
        lines.append(f"CLUSTER {c_id} | {label}")
        lines.append(f"Size: {count} students ({pct:.1f}%)")
        lines.append(f"   [+] Strengths: {str_text}")
        lines.append(f"   [-] Weaknesses: {weak_text}")
        lines.append(f"   -> AI Recommendation: {ai_recommendation}")
        lines.append("-" * 58 + "\n")
        
        print(f"Processed Cluster {c_id}... (Waiting 13s)")
        
        # HIZ LİMİTİ KORUMASI (5 RPM = ~12 sn bekleme)
        if not DEMO_MODE: 
            time.sleep(13) 

    # Kaydet
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines([l + "\n" for l in lines])
        
    print(f"\nAnalysis complete! Report saved to: {REPORT_PATH}")
    print("Check the file to see the Gemini generated advice.")

if __name__ == "__main__":
    generate_cluster_report()