import streamlit as st
from PIL import Image
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile
import os

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Multimodal AI Agent", layout="centered")

# Beautiful Header
st.markdown("""
<h1 style='text-align: center; color: #2c3e50;'>📸 Multimodal AI Agent</h1>
<h4 style='text-align: center; color: #7f8c8d;'>Upload an image, and let the AI analyze it. Extract text, generate descriptions, and get insights instantly.</h4>
<hr>
""", unsafe_allow_html=True)

# --- LOAD AI MODELS ---
@st.cache_resource
def load_models():
    with st.spinner("🔄 Loading AI Models... Please wait (this takes about 1 minute on the first load)."):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        reader = easyocr.Reader(['en'])
    return processor, model, reader

processor, model, reader = load_models()

# --- UI INPUTS ---
uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png", "webp"])

# User question input
user_question = st.text_input("💬 Ask a question about this image (Optional):", placeholder="e.g. What text is in this document? or Describe the scene...")

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # --- PROCESS THE IMAGE ---
    with st.spinner("🧠 AI is analyzing the image..."):
        # 1. Vision AI (BLIP-2)
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        image_desc = processor.decode(out[0], skip_special_tokens=True)
        
        # 2. OCR (EasyOCR)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            image.save(tmpfile.name)
            extracted_text = " ".join(reader.readtext(tmpfile.name, detail=0))
        os.unlink(tmpfile.name)
        
    # --- DISPLAY RESULTS ---
    st.success("✅ Analysis Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI Description")
        st.write(image_desc)
        
    with col2:
        st.subheader("📝 Extracted Text (OCR)")
        if extracted_text.strip():
            st.code(extracted_text, language="text")
        else:
            st.write("No text detected in this image.")
    
    # --- ANSWER THE USER'S QUESTION ---
    if user_question:
        st.subheader("💡 AI Analysis")
        # Simple logic to combine the user's question with the extracted data
        response = f"Based on the image I analyzed:\n\n**Visual Description:** {image_desc}\n\n**Text Found:** {extracted_text[:500]}\n\nTo answer your question: '{user_question}' - I have provided the image description and extracted text above for your review."
        st.info(response)
