
import streamlit as st
from PIL import Image
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import easyocr
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import tempfile
import os

st.set_page_config(page_title="AI RAG Agent", layout="centered")
st.title("🤖 Multimodal AI Agent")
st.write("Upload an image and ask a question. The AI will read the image and answer you.")

# Load models on CPU
@st.cache_resource
def load_models():
    with st.spinner("Loading AI Models..."):
        # BLIP
        v_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        v_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # EasyOCR (CPU)
        o_reader = easyocr.Reader(['en'], gpu=False)
        
        # LLM (Phi-3 on CPU)
        l_generator = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", device_map="cpu")
        
        # Embedding (CPU)
        e_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        dim = 384
        idx = faiss.IndexFlatL2(dim)
        
    return v_processor, v_model, o_reader, l_generator, e_model, idx

vision_processor, vision_model, ocr_reader, llm_generator, embedder, index = load_models()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
user_question = st.text_input("Ask a question about this image", placeholder="e.g. What is this image about?")

if uploaded_file is not None and user_question:
    if st.button("Analyze with AI"):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")
        
        with st.spinner("AI is thinking..."):
            # 1. Vision AI
            inputs = vision_processor(image, return_tensors="pt")
            out = vision_model.generate(**inputs)
            image_desc = vision_processor.decode(out[0], skip_special_tokens=True)
            
            # 2. OCR
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                uploaded_file.seek(0)
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            ocr_result = ocr_reader.readtext(tmp_path, detail=0)
            extracted_text = " ".join(ocr_result)
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
            # 3. RAG
            if extracted_text.strip():
                vector = embedder.encode([extracted_text])
                index.add(vector)
                query_vec = embedder.encode([user_question])
                D, I = index.search(query_vec, 1)
                context = extracted_text
            else:
                context = "No text found in image."
            
            # 4. Generate Answer
            prompt = f"""
            You are an expert AI assistant analyzing an image.
            Context from Image description: {image_desc}
            Context from Text in Image: {context}
            User Question: {user_question}
            Provide a detailed, accurate answer to the user based ONLY on the image and text provided above.
            """
            result = llm_generator(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
            
        st.success("Analysis Complete!")
        st.subheader("🤖 AI Answer")
        st.write(result)
