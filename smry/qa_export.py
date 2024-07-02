import os
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import logging
import warnings
import google.generativeai.types.generation_types
import time

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def answer_question_from_file(file_path, user_question):
    if file_path.endswith(".pdf"):
        text = get_text_from_pdf(file_path)
    elif file_path.endswith(".txt"):
        text = get_text_from_text_file(file_path)
    else:
        return "Unsupported file type."

    if not text:
        return "Error: Unable to extract text from the file."

    text_chunks = get_text_chunks(text)
    if not text_chunks:
        return "Error: Failed to split text into chunks."

    vector_store = get_vector_store(text_chunks)
    if not vector_store:
        return "Error: Failed to create vector store."

    return user_input(user_question, vector_store)

def get_text_from_pdf(pdf_file):
    text = ""
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    except FileNotFoundError:
        logging.error(f"File '{pdf_file}' not found.")
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")
    return text

def get_text_from_text_file(text_file):
    text = ""
    try:
        with open(text_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        logging.error(f"File '{text_file}' not found.")
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not embeddings:
        logging.error("Failed to generate embeddings.")
        return None
    
    if not text_chunks:
        logging.error("No text chunks provided.")
        return None
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    if not vector_store:
        logging.error("Failed to create vector store.")
        return None
    
    vector_store.save_local("faiss_index")
    return vector_store

def user_input(user_question, vector_store):
    prompt_template = """
    You are an AI assistant that provides helpful answers to user queries.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer: 
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    search = vector_store.similarity_search(user_question)

    response = chain(
        {"input_documents": search, "question": user_question},
        return_only_outputs=True
    )

    if response:
        answer_text = response['output_text']
        logging.info(f"\nQuestion: {user_question}, \n")
        logging.info(f"\nAnswer: {answer_text}, \n")
    else:
        answer_text = "Not found in document."
        logging.warning(f"Question: {user_question}, Answer: {answer_text}")

    return clean_text(answer_text)

def clean_text(output_text):
    cleaned_text = output_text.strip()
    cleaned_text = '\n'.join(line for line in cleaned_text.splitlines() if line.strip())
    return cleaned_text

def answer_question_from_file_with_retry(file_path, user_question):
    retry_attempts = 3
    current_attempt = 1
    while current_attempt <= retry_attempts:
        try:
            return answer_question_from_file(file_path, user_question)
        except google.generativeai.types.generation_types.StopCandidateException as e:
            print("An error occurred: StopCandidateException")
            return "An error occurred: StopCandidateException"
        except Exception as e:
            print(f"An error occurred: {e}")
            if current_attempt == retry_attempts:
                return "An error occurred: Maximum retry attempts reached"
            else:
                wait_time = 2 ** current_attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                current_attempt += 1

def main():
    # Specify the file path and user question here
    file_path = "/Users/macbook/Desktop/llm-deployment/data/243.txt"  # Update with your file path
    user_question = "this document is about"  # Update with your question

    try:
        answer = answer_question_from_file_with_retry(file_path, user_question)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
