#%% packages
from langchain.document_loaders import Docx2txtLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
#%% Embeddings Model
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# %%
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-minilm-l6-v2')

def tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 10,
    length_function = tokens,
    add_start_index = True,
)
# %%
loader = Docx2txtLoader("../data/Vector Databases.docx")
pages = loader.load_and_split()

#%% Split the text with text_splitter
docs_texts = text_splitter.split_documents(pages)
# %%
texts = [doc.page_content for doc in docs_texts]
# %%
embeddings = embeddings_model.embed_documents(texts)
# %% for each chunk, calculate the embeddings
len(embeddings)
# %% get the length of vectors
len(embeddings[0])

# %%
