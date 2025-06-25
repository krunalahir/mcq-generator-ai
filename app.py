import streamlit as st  
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from keybert import KeyBERT
import pandas as pd
import tempfile
import random
import torch
from langchain.embeddings import SentenceTransformerEmbeddings

@st.cache_resource(show_spinner="Loading model...")
def load_embedder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

embedder = load_embedder()
@st.cache_resource 
def load_langchain_embedder():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_model = load_langchain_embedder()
  
kw_model = KeyBERT("all-MiniLM-L6-v2")

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=128)
llm = HuggingFacePipeline(pipeline=pipe)


question_prompt = PromptTemplate(
    input_variables=["context", "answer"],
    template="""
Generate a question based on the following context and answer:
Context: {context}
Answer: {answer}
Question:
"""
)

st.title("📘 MCQ Generator with RAG & CSV Export")
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
answers_input = st.text_area("✍️ Enter answers (comma-separated)")
generate_btn = st.button("🔍 Generate MCQs")

if uploaded_file and answers_input and generate_btn:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    answers = [a.strip() for a in answers_input.split(",") if a.strip()]
    all_mcqs = []

    for answer in answers:
        retrieved_docs = retriever.get_relevant_documents(answer)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        question_chain = LLMChain(llm=llm, prompt=question_prompt)
        question = question_chain.run({"context": context, "answer": answer})

        full_text = "\n\n".join([doc.page_content for doc in documents])
        keyphrases = kw_model.extract_keywords(full_text, top_n=30, stop_words='english')

        answer_embedding = embedder.encode(answer, convert_to_tensor=True)
        candidates = []
        for phrase, _ in keyphrases:
            if phrase.lower() == answer.lower():
                continue
            phrase_embedding = embedder.encode(phrase, convert_to_tensor=True)
            similarity = util.cos_sim(answer_embedding, phrase_embedding).item()
            if 0.3 <= similarity <= 0.75:
                candidates.append((phrase, similarity))

        candidates = sorted(candidates, key=lambda x: -x[1])
        distractors = [phrase for phrase, _ in candidates[:3]]

        for i in range(len(answers)):
    
    # Ensure distractors has at least 3 options
            if len(distractors) < 3:
                st.warning(f"Only {len(distractors)} distractors generated. Filling with placeholders.")
                while len(distractors) < 3:
                    distractors.append("Placeholder")

    # Combine answer with top 3 distractors
            options = [answer] + distractors[:3]
            random.shuffle(options)

    # Store result safely
            result = {
                "answer": answer,
                "question": question,
                "option_1": options[0],
                "option_2": options[1],
                "option_3": options[2],
                "option_4": options[3],
                "correct_option": f"option_{options.index(answer) + 1}"
            }

            all_mcqs.append(result)


    st.subheader("✅ Generated MCQs")
    for i, row in enumerate(all_mcqs, 1):
        st.markdown(f"**{i}. {row['question']}**")
        st.markdown(f"A. {row['option_1']}  B. {row['option_2']}  C. {row['option_3']}  D. {row['option_4']}")
        st.markdown(f"✅ Correct Answer: {row['answer']}")
        st.markdown("---")

    df = pd.DataFrame(all_mcqs)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download MCQs as CSV", data=csv, file_name="mcqs.csv", mime="text/csv")
    st.success("All MCQs generated and saved!")
