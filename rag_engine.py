from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


class PDFRagEngine:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        embed_model: str = "nomic-embed-text-v1.5",
        llm_model: str = "gemma-3-1b-it",
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.embed_model = embed_model
        self.llm_model = llm_model
        
        self.retriever = None
        self.active_filename = None

    def _get_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.embed_model,
            check_embedding_ctx_length=False,
        )

    def _get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.llm_model,
            temperature=0.2,
        )

    def index_pdf(self, file_path: str, original_filename: str) -> int:
        """Loads, splits, and indexes a PDF into the vector store."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self._get_embeddings(),
        )

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.active_filename = original_filename
        return len(splits)

    def ask(self, query: str) -> dict:
        """Retrieves relevant chunks and generates an answer using the LLM."""
        if self.retriever is None:
            raise ValueError("No document has been indexed yet.")

        relevant_docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        		Use only the provided context below to answer the question. 
		If the answer is not in the context, say "I do not know based on the provided PDF."

		Context:{context}
		Question: {query}
		Answer:"""

        llm = self._get_llm()
        response = llm.invoke(prompt)
        sources = sorted({doc.metadata.get("page", 0) + 1 for doc in relevant_docs})

        return {
            "answer": response.content,
            "sources": sources,
        }
