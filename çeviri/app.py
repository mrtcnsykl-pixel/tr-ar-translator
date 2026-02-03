import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Sayfa ayarları
st.set_page_config(page_title="TR-AR Çeviri", page_icon="🌍")


# Model yükleme (Önbelleğe alıyoruz ki site her yenilendiğinde bekleme yapmasın)
@st.cache_resource
def load_model():
    model_path = "BAU_Final_Model"  # Klasör isminin doğru olduğundan emin ol
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Web sunucularında genelde GPU (cuda) olmaz, bu yüzden "cpu" kullanıyoruz.
    # Eğer GPU varsa .to("cuda") yapabilirsin.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return tokenizer, model, device


# Arayüz Elemanları
st.title("🇹🇷 Türkçe - Arapça Çeviri 🇦🇪")
st.markdown("Eğitilmiş model kullanılarak yapılan profesyonel çeviri arayüzü.")

try:
    tokenizer, model, device = load_model()

    # Kullanıcı girişi
    text_to_translate = st.text_area("Türkçe Metni Giriniz:", placeholder="Örn: Merhaba, nasılsın?", height=150)

    if st.button("Çevir"):
        if text_to_translate.strip():
            with st.spinner('Çeviri yapılıyor, lütfen bekleyin...'):
                # Çeviri İşlemi
                inputs = tokenizer(text_to_translate, return_tensors="pt", padding=True, truncation=True,
                                   max_length=128).to(device)

                with torch.no_grad():
                    generated_tokens = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=5,
                        early_stopping=True
                    )

                translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

                # Sonucu Göster
                st.subheader("Arapça Sonuç:")
                st.success(translation)
        else:
            st.warning("Lütfen çevirmek istediğiniz bir metin girin.")

except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")
    st.info("BAU_Final_Model klasörünün app.py ile aynı dizinde olduğundan emin olun.")

st.divider()
st.caption("Bu uygulama BAU Final Projesi kapsamında geliştirilmiştir.")