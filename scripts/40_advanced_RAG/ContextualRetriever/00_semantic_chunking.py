#%% Packages (1)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import WikipediaLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from pprint import pprint
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %% Load the article (2) 
ai_article_title = "Artificial_intelligence"
loader = WikipediaLoader(query=ai_article_title, 
                             load_all_available_meta=True, 
                             doc_content_chars_max=1000, 
                             load_max_docs=1)
doc = loader.load()

# %% check the content (3)
pprint(doc[0].page_content)
# %% Create splitter instance (4)
splitter = SemanticChunker(embeddings=OpenAIEmbeddings(), 
                           breakpoint_threshold_type="percentile", breakpoint_threshold_amount=0.3)

# %% Apply semantic chunking (5)
chunks = splitter.split_documents(doc)

# %% check the results (6)
chunks
# %%
pprint(chunks[0].page_content)
# %%
pprint(chunks[1].page_content)
# %%
text = """
The History of Machine Learning: A Journey from Theory to Everyday Use
Machine learning (ML) has grown from a theoretical field in the mid-20th century to one of the most impactful technologies of the modern age. Its evolution is a story of breakthroughs, challenges, and ever-increasing applications. 1950s – The Foundations Are Laid In the 1950s, the first ideas of machine learning emerged. Alan Turing’s proposal of the "Turing Test" in 1950 marked one of the earliest discussions of machine intelligence. Soon after, in 1956, the term "artificial intelligence" (AI) was officially coined at the Dartmouth Conference, considered the birthplace of AI as a field. Early experiments in this decade primarily focused on rule-based systems and symbolic AI. 1960s–1970s – Early Algorithms and Setbacks The 1960s and 70s saw the development of algorithms like decision trees and nearest neighbor. However, progress slowed in the late 1970s, often referred to as the "AI Winter," due to funding cuts and limitations in computing power. Despite these setbacks, essential research laid the groundwork for future advancements. 1980s – Neural Networks and Renewed Interest A resurgence in interest occurred in the 1980s, driven by the development of neural networks. In 1986, Geoffrey Hinton popularized backpropagation, allowing neural networks to be trained more efficiently. This advancement rekindled enthusiasm, although practical applications were still limited by technology at the time. 1990s–2000s – From Research to Real-World Applications By the 1990s, machine learning had begun to find practical applications, particularly in areas like speech recognition and handwriting recognition. In 1997, IBM’s Deep Blue defeated chess grandmaster Garry Kasparov, a landmark moment showcasing the potential of AI in specific tasks. The rise of the internet and increased data availability in the 2000s further spurred machine learning research. 2010s – The Rise of Deep Learning The 2010s marked the advent of deep learning, transforming fields such as image recognition, natural language processing, and autonomous vehicles. In 2012, a deep neural network trained by Google recognized cats in YouTube videos without any human labeling. This era saw the introduction of frameworks like TensorFlow and PyTorch, enabling developers and researchers to build complex ML models more easily. 2020s – Machine Learning in Everyday Life Today, machine learning is embedded in everyday life, from personal assistants to recommendation systems. With the rise of ethical concerns and the demand for transparency, research has expanded to include explainable AI, model fairness, and privacy-preserving techniques. Conclusion: The Future of Machine Learning; As we look to the future, machine learning promises even more transformative impacts, especially with advances in fields like quantum computing and neuromorphic engineering. However, with these advancements come questions about ethics, regulation, and societal implications.
"""

#%%
splitter = SemanticChunker(embeddings=OpenAIEmbeddings(), 
                           breakpoint_threshold_type="percentile",
                           breakpoint_threshold_amount=0.5)

chunks = splitter.split_text(text)
# %%
len(chunks)
# %%
print(chunks[0])

# %%
print(chunks[1])

#%%
print(chunks[2])

# %%
