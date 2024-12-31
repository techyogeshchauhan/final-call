import fitz
from PIL import Image
import io
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import tempfile
import streamlit as st

class BaseRAG:
    def __init__(self):
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Initializing models...")
        # Move models to GPU
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model.to(self.device)

        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_model.to(self.device)

        self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
        self.chat_model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

        self.text_chunks = []
        self.text_embeddings = []
        self.images = []
        self.image_embeddings = []
        self.image_locations = []
        self.page_text_map = {}
        print("Initialization complete!")

    def chunk_text(self, text, chunk_size=500, overlap=100):
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            chunk = ' '.join(words[start:start + chunk_size])
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def process_text(self, text, page_num):
        if not text.strip():
            return

        if page_num not in self.page_text_map:
            self.page_text_map[page_num] = []

        chunks = self.chunk_text(text)
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                self.text_chunks.append(chunk)
                self.page_text_map[page_num].append(chunk)
                embedding = self.text_model.encode(chunk, convert_to_tensor=True)
                embedding = embedding.cpu().numpy()
                self.text_embeddings.append(embedding)

    def extract_content_from_pdf(self, pdf_path):
        print(f"Processing PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]

            # Extract text
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4]
                self.process_text(text, page_num)

            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')

                    inputs = self.image_processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        image_features = self.image_model.get_image_features(**inputs)

                    self.images.append(image)
                    self.image_embeddings.append(image_features.cpu().numpy())
                    self.image_locations.append((page_num, img_index))
                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_num}: {str(e)}")
                    continue

        doc.close()

        if self.text_embeddings:
            self.text_embeddings = np.vstack(self.text_embeddings)

        print(f"Processing complete! Found {len(self.text_chunks)} text chunks and {len(self.images)} images.")

    def search(self, query, top_k=3):
        results = {
            'text': [],
            'images': [],
            'text_locations': [],
            'image_locations': [],
            'context': []
        }

        if len(self.text_chunks) > 0:
            text_query_embedding = self.text_model.encode(query, convert_to_tensor=True)
            text_query_embedding = text_query_embedding.cpu().numpy()

            text_similarities = cosine_similarity(
                [text_query_embedding],
                self.text_embeddings
            )[0]

            top_text_indices = np.argsort(text_similarities)[-top_k:][::-1]
            results['text'] = [self.text_chunks[i] for i in top_text_indices]
            results['text_locations'] = top_text_indices

            for idx in top_text_indices:
                page_num = None
                for p, chunks in self.page_text_map.items():
                    if self.text_chunks[idx] in chunks:
                        page_num = p
                        break

                if page_num is not None:
                    context = "\n".join(self.page_text_map[page_num])
                    results['context'].append((context, page_num))

        if len(self.images) > 0:
            image_query_inputs = self.image_processor(text=[query], return_tensors="pt", padding=True)
            image_query_inputs = {k: v.to(self.device) for k, v in image_query_inputs.items()}

            with torch.no_grad():
                image_query_features = self.image_model.get_text_features(**image_query_inputs)

            image_query_features = image_query_features.cpu().numpy()

            image_similarities = cosine_similarity(
                image_query_features,
                np.vstack(self.image_embeddings)
            )[0]

            top_image_indices = np.argsort(image_similarities)[-top_k:][::-1]
            results['images'] = [self.images[i] for i in top_image_indices]
            results['image_locations'] = [self.image_locations[i] for i in top_image_indices]

        return results

    def generate_response(self, query, context):
        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {query}

Answer: """

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.chat_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer: ")[-1].strip()

class EnhancedRAG(BaseRAG):
    def __init__(self):
        super().__init__()
        self.headings = {}
        self.toc = []
        self.pdf_display = None

    def extract_headings(self, pdf_path):
        doc = fitz.open(pdf_path)
        current_heading = "Introduction"
        current_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            font_size = span["size"]
                            is_bold = span["flags"] & 2 ** 4
                            
                            if (font_size > 12 or is_bold) and text and len(text) < 100:
                                if current_heading and current_content:
                                    self.headings[current_heading] = "\n".join(current_content)
                                current_heading = text
                                current_content = []
                                self.toc.append((current_heading, page_num + 1))
                            else:
                                if text:
                                    current_content.append(text)
        
        if current_heading and current_content:
            self.headings[current_heading] = "\n".join(current_content)
        
        self.pdf_display = doc
        return self.headings, self.toc

    def get_page_image(self, page_num):
        if self.pdf_display:
            page = self.pdf_display[page_num]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data))
        return None

# [Previous code remains the same until the display_pdf_content method in StreamlitRAG class]

class StreamlitRAG(EnhancedRAG):
    def display_results(self, results):
        if results['text']:
            st.subheader("Relevant Text Contexts:")
            for i, (text, location) in enumerate(zip(results['text'], results['text_locations']), 1):
                with st.expander(f"Text Chunk {i}"):
                    st.write(text)

        if results['images']:
            st.subheader("Relevant Images:")
            cols = st.columns(min(3, len(results['images'])))
            for i, (img, loc, col) in enumerate(zip(results['images'], 
                                                  results['image_locations'], 
                                                  cols)):
                with col:
                    st.image(img, caption=f"Image from page {loc[0] + 1}")

    def display_pdf_content(self):
        st.sidebar.title("Table of Contents")
        
        # Create a unique key for each button using the heading and page number
        for heading, page_num in self.toc:
            button_key = f"toc_button_{heading}_{page_num}"  # Create unique key
            if st.sidebar.button(f"{heading} (Page {page_num})", key=button_key):
                st.session_state.selected_heading = heading
                st.session_state.current_page = page_num - 1

        col1, col2 = st.columns([3, 2])
        
        with col1:
            if hasattr(st.session_state, 'selected_heading'):
                st.subheader(st.session_state.selected_heading)
                content = self.headings.get(st.session_state.selected_heading, "")
                st.write(content)
        
        with col2:
            if hasattr(st.session_state, 'current_page'):
                page_image = self.get_page_image(st.session_state.current_page)
                if page_image:
                    st.image(page_image, caption=f"Page {st.session_state.current_page + 1}")

# [Rest of the code remains the same]

def main():
    st.set_page_config(page_title="Multimodal RAG System", 
                      page_icon="🤖",
                      layout="wide")

    st.title("📚 Multimodal RAG System")

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
        st.session_state.pdf_processed = False
        st.session_state.selected_heading = None
        st.session_state.current_page = 0

    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

    if uploaded_file:
        if not st.session_state.pdf_processed:
            with st.spinner("Initializing RAG system and processing PDF..."):
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                st.session_state.rag_system = StreamlitRAG()
                st.session_state.rag_system.extract_headings(temp_path)
                st.session_state.rag_system.extract_content_from_pdf(temp_path)
                
                st.session_state.pdf_processed = True
                st.success("PDF processed successfully!")

        qa_col, content_col = st.columns([1, 1])

        with qa_col:
            st.subheader("Ask questions about your document")
            query = st.text_input("Enter your question:")

            if query:
                with st.spinner("Generating response..."):
                    results = st.session_state.rag_system.search(query)
                    context = ""
                    if results['context']:
                        context = "\n".join([ctx[0] for ctx in results['context']])
                    
                    response = st.session_state.rag_system.generate_response(query, context)
                    st.write("### Answer:")
                    st.write(response)
                    st.session_state.rag_system.display_results(results)

        with content_col:
            st.subheader("Document Content")
            st.session_state.rag_system.display_pdf_content()

    else:
        st.info("Please upload a PDF document to begin.")
        st.session_state.rag_system = None
        st.session_state.pdf_processed = False
        st.session_state.selected_heading = None
        st.session_state.current_page = 0

if __name__ == "__main__":
    main()