#%% packages
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
# %%
loader = Docx2txtLoader("../data/Vector Databases.docx")
pages = loader.load_and_split()
print(f'Loaded {len(pages)} pages from the Word Document.')

#%%
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-minilm-l6-v2')

def tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 10,
    length_function = tokens,
    add_start_index = True,
)

# %% aggregate the chunks to a certain size
def aggregate_documents_by_tokens(documents, target_token_size, tokenizer):
    aggregated_documents = []
    current_tokens = []
    current_metadata = {}

    for doc in documents:
        doc_tokens = tokenizer(doc.page_content, add_special_tokens=False)["input_ids"]
        
        if len(current_tokens) + len(doc_tokens) <= target_token_size:
            current_tokens.extend(doc_tokens)
            # Combine metadata (optional: you might want to customize this part)
            current_metadata.update(doc.metadata)
        else:
            # Create a new Document with aggregated content and metadata
            aggregated_documents.append(Document(
                page_content=tokenizer.decode(current_tokens, clean_up_tokenization_spaces=True),
                metadata=current_metadata
            ))
            current_tokens = doc_tokens
            current_metadata = doc.metadata
    
    # Append the last document if it's not empty
    if current_tokens:
        aggregated_documents.append(Document(
            page_content=tokenizer.decode(current_tokens, clean_up_tokenization_spaces=True),
            metadata=current_metadata
        ))
    
    return aggregated_documents


# %% Split the text with text_splitter
texts = text_splitter.split_documents(pages)
print(f'Number of Chunks after splitting: {len(texts)}')
# get the number of tokens in each chunk
chunks = [tokens(doc.page_content) for doc in texts]
print(f"Number of Tokens in each chunk: {chunks}")

#%% visualize number of tokens in each chunk as barplot
plt.bar(range(len(chunks)), chunks)
plt.title('Number of Tokens in each chunk')


#%% aggregate documents by tokens
docs_aggregated = aggregate_documents_by_tokens(documents=texts, target_token_size=100, tokenizer=tokenizer)
# %% get the size of the aggregated documents
agg_chunks = [tokens(doc.page_content) for doc in docs_aggregated]
print(f'Number of Chunks after aggregation: {len(agg_chunks)}')
# get the number of tokens in each chunk
print(f"Number of Tokens in each chunk: {agg_chunks}")
# %% visualize number of tokens in each chunk as barplot
plt.bar(range(len(agg_chunks)), agg_chunks)
plt.title('Number of Tokens in each chunk')

# %%
