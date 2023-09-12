import chromadb
import structlog as logging

logger = logging.get_logger(__name__)


class PromptBuilder:

    def get_complete_prompt(self, query: str, chroma_client: chromadb.API) -> str:
        """
        Build a model prompt string given a user query and vector DB of documents
        """
        documents = self._get_context_for_query(query, chroma_client)
        prompt = self._build_prompt_with_context(documents, query)
        logger.info(f"Created prompt: {prompt}")

        return prompt

    def _get_context_for_query(self, query: str, chroma_client: chromadb.API) -> []:
        # search the chroma db for document chunks that match the user query
        results = chroma_client.get_collection("all-documents").query(
            query_texts=[query],
            n_results=2,
        )
        logger.info(f"Retrieved matching documents from Chroma: {results}")
        documents = results['documents']

        if documents is None:
            return []
        return documents[0]

    def _build_prompt_with_context(self, documents: [], query: str) -> str:
        CONTEXT_PROMPT = "Given this document: "
        LOCAL_LLAMA_PROMPT = "You are an artificial intelligence striving towards providing the most accurate and " \
                             "comprehensive answers possible.\n Q: "
        return f"{CONTEXT_PROMPT} {documents} {LOCAL_LLAMA_PROMPT} {query} A: "
