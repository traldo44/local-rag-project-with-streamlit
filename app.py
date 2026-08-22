import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import sqlite3
import json
import numpy as np
import PyPDF2
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Yerel İzahname Özetleyicisi", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-image: radial-gradient(#222 1px, transparent 1px);
        background-size: 32px 32px;
    }
    
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stChatInputContainer {
        border-radius: 12px;
        border: 1px solid #333 !important;
        background-color: #111 !important;
        box-shadow: 0 0 20px rgba(249, 115, 22, 0.1);
        transition: all 0.3s ease;
    }
    .stChatInputContainer:focus-within {
        border-color: #f97316 !important;
        box-shadow: 0 0 30px rgba(249, 115, 22, 0.25);
    }
    
    [data-testid="stExpander"] {
        background-color: rgba(23, 23, 23, 0.7);
        border: 1px solid #333;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] {
        border-right: 1px solid #222;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    button[kind="header"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

DB_NAME = "rag_database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  source_file TEXT, 
                  text_content TEXT, 
                  embedding TEXT)''')
    conn.commit()
    conn.close()

init_db()

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource(show_spinner=False)
def load_llm():
    return pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")

embedder = load_embedder()
llm = load_llm()


# <İŞARET>: BURADA CHUNK BOYUTU AYARLANIYOR
def get_chunks(text, chunk_size=1200, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_pdfs(uploaded_files):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('DELETE FROM documents')
    
    for file in uploaded_files:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
                
        # <İŞARET>: BURADA CHUNK BOYUTU AYARLANIYOR
        chunks = get_chunks(text, chunk_size=1200, overlap=300)
        for chunk in chunks:
            # <İŞARET>: BURADA METADATA EKLENİYOR
            final_chunk = f"[Kaynak: {file.name}]\n{chunk}"
            emb = embedder.encode(final_chunk).tolist()
            c.execute('INSERT INTO documents (source_file, text_content, embedding) VALUES (?, ?, ?)', 
                      (file.name, final_chunk, json.dumps(emb)))
    
    conn.commit()
    conn.close()

with st.sidebar:
    st.title("📊 İzahname Yönetimi")
    st.write("Sisteme eklenecek Finansal İzahname (PDF) dosyalarını seçin.")
    
    uploaded_files = st.file_uploader("İzahname Yükle", type=["pdf"], accept_multiple_files=True)
    
    if st.button("İzahnameleri İşle", use_container_width=True):
        if uploaded_files:
            with st.status("Belgeler sisteme entegre ediliyor...", expanded=True) as status:
                st.write("📄 PDF'ler okunuyor ve parçalara ayrılıyor...")
                process_pdfs(uploaded_files)
                st.write("🧠 Yapay zeka vektörleri oluşturuldu ve kaydedildi...")
                status.update(label="İşlem tamamlandı!", state="complete", expanded=False)
            
            st.markdown("<h3 style='color: #f97316; text-align: center; margin-top: 10px;'>✔ Done</h3>", unsafe_allow_html=True)
            st.toast("Veritabanı güncellendi!")
        else:
            st.error("Lütfen en az bir dosya seçin.")
            
    st.divider()
    st.caption("Yerel İzahname Özetleyicisi | SQLite & Qwen2.5")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def search_db(query, top_k=2):
    query_emb = embedder.encode(query).tolist()
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT text_content, embedding FROM documents')
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        content = row[0]
        emb = json.loads(row[1])
        sim = cosine_similarity(query_emb, emb)
        results.append((sim, content))
        
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

st.markdown("""
<div style='margin-top: 3rem; margin-bottom: 4rem;'>
    <h1 style='font-size: 4rem; font-weight: 800; line-height: 1.1; letter-spacing: -2px; margin-bottom: 1.5rem; color: white;'>
        Yükle,<br>
        sorgula, analiz et,<br>
        <span style='color: #f97316;'>tekrarla.</span>
    </h1>
    <p style='font-size: 1.25rem; color: #888; max-width: 650px; line-height: 1.6;'>
        <b>Yerel İzahname Özetleyicisi</b> tamamen bilgisayarınızda çalışır. Verileriniz güvendedir.
        Finansal dokümanlarınızı yükleyin ve Qwen2.5 yapay zekasıyla saniyeler içinde analiz edin.
    </p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar_icon = "💼" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "context" in msg:
            with st.expander("Eşleşen Kaynak Metinleri Görüntüle"):
                st.write(msg["context"])

if prompt := st.chat_input("İzahnamelerde aramak istediğiniz bilgiyi yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        
    with st.chat_message("assistant", avatar="💼"):
        with st.status("Sistem izahnameleri tarıyor...", expanded=True) as status:
            st.write("🔎 Veritabanında eşleşen parçalar aranıyor...")
            top_results = search_db(prompt, top_k=2)
            
            if not top_results:
                status.update(label="Eşleşme bulunamadı.", state="error", expanded=False)
                response_text = "İzahnameler içerisinde aradığınız bilgiye dair bir eşleşme bulunamadı."
                context_joined = None
            else:
                st.write("🧠 Qwen2.5 (AI) ile özet sentezleniyor...")
                context_texts = [res[1] for res in top_results]
                context_joined = "\n\n---\n\n".join(context_texts)
                
                messages = [
                    {"role": "system", "content": "Sen net ve ciddi bir finans asistanısın. Sadece aşağıdaki 'Bilgi' metninde yazanları kullanarak cevap ver. Kendi fikrini katma, uydurma. Eğer bilgi metninde cevap yoksa sadece 'Bu bilgi izahnamede bulunmuyor.' de."},
                    {"role": "user", "content": f"Bilgi:\n{context_joined}\n\nSoru: {prompt}"}
                ]
                
                output = llm(
                    messages, 
                    max_new_tokens=512, 
                    return_full_text=False,
                    do_sample=False, 
                    repetition_penalty=1.15
                )
                
                try:
                    generated = output[0]["generated_text"]
                    if isinstance(generated, list):
                        response_text = generated[-1]["content"]
                    else:
                        response_text = generated
                except Exception as e:
                    response_text = str(output)
                
                status.update(label="Yanıt başarıyla oluşturuldu! ✔", state="complete", expanded=False)
                
        st.markdown(response_text)
        
        if context_joined:
            with st.expander("Eşleşen Kaynak Metinleri Görüntüle"):
                st.write(context_joined)
                
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text, 
                "context": context_joined
            })
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text
            })
