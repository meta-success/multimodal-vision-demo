
import streamlit as st
from PIL import Image
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile
import os

st.set_page_config(page_title="Multimodal Vision & OCR", layout="centered")
st.title("📸 Multimodal AI Agent")
st.write("Upload an image and watch the AI analyze it!")

@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    reader = easyocr.Reader(['en'])
    return processor, model, reader

processor, model, reader = load_models()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("AI is analyzing the image..."):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        image_desc = processor.decode(out[0], skip_special_tokens=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            image.save(tmpfile.name)
            extracted_text = " ".join(reader.readtext(tmpfile.name, detail=0))
        os.unlink(tmpfile.name)
        
    st.success("Analysis Complete!")
    st.subheader("🤖 AI Description")
    st.write(image_desc)
    st.subheader("📝 Extracted Text (OCR)")
    if extracted_text.strip():
        st.code(extracted_text, language="text")
    else:
        st.write("No text detected in this image.")
