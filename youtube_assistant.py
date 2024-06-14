from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv
# import os
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore

load_dotenv()

def vector_db_from_video(video_url):
    # Initialize embedding with ADA model
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # # Initialize Pinecone
    # pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # pinecone_environment = os.getenv("PINECONE_ENV")
    #
    # pc = Pinecone(api_key=pinecone_api_key)
    #
    # # Create Pinecone index
    # index_name = "youtube-transcripts"
    #
    # if index_name not in pc.list_indexes().names():
    #     pc.create_index(name=index_name, dimension=1536,
    #                     metric="dotproduct",
    #                     spec=ServerlessSpec(cloud="aws", region="us-east-1")
    #                     )
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(transcript)
    # vector_store = PineconeVectorStore.from_documents(documents=documents, embedding=embeddings,
    #                                                   index_name=index_name)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


# print(vector_db_from_video("https://youtu.be/uElaHUSM7fI"))

def get_response_from_query(vector_store, query, k=4):
    docs = vector_store.similarity_search(query, k=k)
    docs_page_content = ''.join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    template = """
    You are a helpful YouTube assistant that can answer a question about a video based on that video's transcript.
    Answer the following question: {question}
    By searching the following video transcript: {docs}
    
    Only use factual information from the transcript to answer the question.
    If you feel like you do not have enough information to answer the question, say "I don't know."
    
    Give detailed answers.
    """

    prompt = PromptTemplate(input_variables=['question', 'docs'], template=template)
    formatted_prompt = prompt.format(question=query, docs=docs_page_content)

    response = llm.invoke(formatted_prompt)

    return response.content

# vector_store = vector_db_from_video("https://www.youtube.com/watch?v=lG7Uxts9SXs")
# query = "What is OpenAI?"
# response = get_response_from_query(vector_store, query, 4)
#
# print(response)
