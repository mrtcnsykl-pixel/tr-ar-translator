import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Sayfa tasarımı
st.set_page_config(page_title="TR-AR Çeviri", page_icon="🌐")

@st.cache_resource
def load_model():
    # Senin Hugging Face model kimliğin
    model_id = "Saykal/tr-ar-translator-model" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ücretsiz sunucularda GPU olmadığı için CPU kullanıyoruz
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cpu")
    return tokenizer, model

st.title("🇹🇷 Türkçe - Arapça Çeviri 🇦🇪")
st.markdown("Hugging Face üzerinden çalışan yapay zeka modeli.")

try:
    tokenizer, model = load_model()
    
    user_input = st.text_area("Çevrilecek metni yazın:", height=100)

    if st.button("Çevir"):
        if user_input.strip():
            with st.spinner('Çevriliyor...'):
                inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    generated_tokens = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=5
                    )
                translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                st.subheader("Sonuç:")
                st.success(translation)
        else:
            st.warning("Lütfen bir metin girin.")
except Exception as e:
    st.error(f"Hata oluştu: {e}")
