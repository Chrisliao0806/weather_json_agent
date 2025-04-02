import os
import logging
import json
import requests
from langchain.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import END, START, StateGraph
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.llm_usage import get_llm
from utils.process_json import process_weather_json
from utils.logger import setup_logging
from utils.choose_state import State, GradeDocuments
from utils.prompt import (
    INSTRUCTIONRAGGRADE,
    INSTRUCTIONRAG,
    INSTRUCTIONPLAIN,
)


EMBEDDING = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "mps"},
)


class WeatherAgent:
    """
    WeatherAgent is a class designed to handle weather data retrieval, processing, and 
    retrieval-augmented generation (RAG) workflows. It integrates with language models 
    and vector databases to provide intelligent responses based on weather data.

        llm (object): The language model instance used for generating responses.
        file_path (str): Path to the JSON file for storing weather data.
        vectordb (Chroma): The Chroma vector database instance for document embeddings.
        rag_chain (object): The RAG chain for generating responses using retrieved documents.
        llm_chain (object): The plain LLM chain for generating responses without retrieval.
        retrieval_grader (object): The chain for grading document relevance.

    Methods:
        __init__(chroma_file=None, file_path=None, use_local_llm=True):
            Initializes the WeatherAgent instance with optional Chroma file, file path, 
            and local LLM usage.

        _init_model():
            Initializes the language model chains and retrieval grader.

        get_weather_data():
            Retrieves weather data from an external API and processes it into a JSON file.

        get_weather_data_from_file(file_path):
            Reads and processes weather data from a local JSON file.

        document_embedding():
            Generates document embeddings using a sentence-transformers model and manages 
            a Chroma vector database for retrieval tasks.

        retrieve(state):
            Retrieves relevant documents based on the given question in the state.

        retrieval_grade(state):
            Filters retrieved documents based on their relevance to the question.

        route_retrieval(state):
            Determines whether to generate an answer using RAG or plain feedback.

        rag_generate(state):
            Generates a response in RAG (Retrieval-Augmented Generation) mode using 
            retrieved documents.

        plain_generate(state):
            Generates a response in plain mode using only the question.

        workflow(query):
            Executes the workflow for processing a query using a state graph, including 
            retrieval, grading, conditional routing, and response generation.

        ValueError: If weather data retrieval fails.
        FileNotFoundError: If the specified local JSON file does not exist.
    """
    def __init__(self, chroma_file=None, file_path=None, use_local_llm=True):
        # Initialize the LLM
        self.llm = get_llm(local_llm=use_local_llm, model_name="qwen2.5:7b")
        self.chroma_file = chroma_file
        self.file_path = file_path
        self.document_embedding()
        self.rag_chain, self.llm_chain, self.retrieval_grader = self._init_model()

    def _init_model(self):
        """
        Initializes the language model used for generating SQL queries and answers.
        """
        prompt_rag = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONRAG),
                ("system", "文件: \n\n {documents}"),
                ("human", "問題: {question}"),
            ]
        )

        prompt_plain = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONPLAIN),
                ("human", "問題: {question}"),
            ]
        )
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONRAGGRADE),
                ("human", "文件: \n\n{documents}"),
                ("human", "問題: {question}"),
            ]
        )
        # LLM & chain
        rag_chain = prompt_rag | self.llm | StrOutputParser()
        # LLM & chain
        llm_chain = prompt_plain | self.llm | StrOutputParser()

        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        # 使用 LCEL 語法建立 chain
        retrieval_grader = grade_prompt | structured_llm_grader
        return rag_chain, llm_chain, retrieval_grader

    def get_weather_data(self):
        """
        Read and process the weather data from the specified JSON file.
        """
        url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001"
        params = {
            "Authorization": os.getenv("WEB_KEY"),
            "format": "JSON",
        }
        response = requests.get(url, params=params, timeout=10)
        json_file = response.json()
        if not json_file:
            logging.error("Failed to retrieve weather data.")
            raise ValueError("Failed to retrieve weather data.")
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(json_file, f, ensure_ascii=False, indent=2)
        return process_weather_json(json_file, self.file_path)

    def get_weather_data_from_file(self, file_path):
        """
        Read and process the weather data from a local JSON file.
        """
        if not os.path.exists(file_path):
            logging.error("File %s does not exist.", file_path)
            raise FileNotFoundError(f"File {file_path} does not exist.")

        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        documents = process_weather_json(json_data, file_path)
        return documents

    def document_embedding(self):
        """
        Generates document embeddings using the HuggingFace sentence-transformers model
        and manages a Chroma vector database for retrieval tasks.

        This method checks if a Chroma file exists. If it does, it initializes the vector
        database from the existing file. Otherwise, it processes weather data, creates
        embeddings, and persists the vector database to a file. The method also sets up
        a retriever for querying the vector database.

        Attributes:
            chroma_file (str): Path to the Chroma persistence directory.
            vectordb (Chroma): The Chroma vector database instance.
            retriever (Retriever): The retriever instance for querying the vector database.

        Raises:
            Exception: If there is an issue with embedding or database creation.

        Logging:
            Logs whether the Chroma file exists or is created.
        """
        # Embed text
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "mps"},
        )
        if self.chroma_file:
            self.vectordb = Chroma(
                persist_directory=self.chroma_file,
                embedding_function=embedding,
                collection_name="coll2",
                collection_metadata={"hnsw:space": "cosine"},
            )
            logging.info("Chroma file exists")
        else:
            all_splits = self.get_weather_data()
            self.vectordb = Chroma.from_documents(
                documents=all_splits,
                embedding=embedding,
                collection_name="coll2",
                collection_metadata={"hnsw:space": "cosine"},
                persist_directory=self.chroma_file,
            )
            self.vectordb.persist()
            logging.info("Chroma file created")
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

    def retrieve(self, state):
        """
        Retrieve relevant documents based on the given question in the state.

        Args:
            state (dict): A dictionary containing the question to be used for retrieval.

        Returns:
            dict: A dictionary containing:
                - "documents" (list): A list of tuples where each tuple
                                      contains a document and its relevance score.
                - "question" (str): The original question from the state.
                - "use_rag" (bool): A flag indicating whether the relevance score
                                    of any document exceeds the threshold (0.3).
        """

        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        # 0.3 is the threshold for relevance score
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def retrieval_grade(self, state):
        """
        filter retrieved documents based on question.

        Args:
            state (dict):  The current state graph

        Returns:
            state (dict): New key added to state, documents, 
            that contains list of related documents.
        """

        # Grade documents
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

        documents = state["documents"]
        question = state["question"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"documents": d.page_content, "question": question}
            )
            grade = score.binary_score
            if grade == "yes":
                print("  -GRADE: DOCUMENT RELEVANT-")
                filtered_docs.append(d)
            else:
                print("  -GRADE: DOCUMENT NOT RELEVANT-")
                continue
        return {"documents": filtered_docs, "question": question}

    def route_retrieval(self, state):
        """
        Determines whether to generate an answer, or use websearch.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ROUTE RETRIEVAL---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            print(
                "  -DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, ROUTE TO PLAIN ANSWER-"
            )
            return "plain_feedback"
        else:
            # We have relevant documents, so generate answer
            print("  -DECISION: GENERATE WITH RAG LLM-")
            return "rag_generate"

    def rag_generate(self, state):
        """
        Generates a response in RAG (Retrieval-Augmented Generation) mode.

        This method takes a state dictionary containing a question and a list of documents,
        and uses the RAG chain to generate a response based on the provided documents and question.

        Args:
            state (dict): A dictionary containing the following keys:
                - "question" (str): The question to be answered.
                - "documents" (list): A list of documents to be used for generating the response.

        Returns:
            dict: A dictionary containing the original question,
                  documents, and the generated response.
                - "question" (str): The original question.
                - "documents" (list): The original list of documents.
                - "generation" (str): The generated response.
        """
        print("---GENERATE IN RAG MODE---")
        question = state["question"]
        documents = state["documents"]
        # RAG generation
        generation = self.rag_chain.invoke(
            {"documents": documents, "question": question}
        )
        return {"documents": documents, "question": question, "generation": generation}

    def plain_generate(self, state):
        """
        Generates a response in plain mode.

        This method takes a state dictionary containing a question and uses the LLM chain
        to generate a response based solely on the question.
        Args:
            state (dict): A dictionary containing the following keys:
                - "question" (str): The question to be answered.
                - "documents" (list): A list of documents (not used in this method).
        Returns:
            dict: A dictionary containing the original question,
                  documents, and the generated response.
                - "question" (str): The original question.
                - "documents" (list): The original list of documents.
                - "generation" (str): The generated response.
        """
        print("---GENERATE IN PLAIN MODE---")
        question = state["question"]
        # Plain generation
        generation = self.llm_chain.invoke({"question": question})
        return {"question": question, "generation": generation}

    def workflow(self, query):
        """
        Executes the workflow for processing a query using a state graph.

        The workflow consists of the following steps:
        1. Retrieve: Retrieves relevant data or documents.
        2. Grade Documents: Grades the retrieved documents based on relevance.
        3. Conditional Routing: Routes the workflow based on the grading results.
           - If the route is "rag_generate", it proceeds to generate a response 
           using RAG (Retrieval-Augmented Generation).
           - If the route is "plain_feedback", it ends the workflow at the first stage.
        4. RAG Generate: Generates a response using RAG if applicable.
        5. Plain Feedback: Ends the workflow with plain feedback if applicable.

        Args:
            query (str): The input query to process.

        Returns:
            tuple: A tuple containing:
                - output (dict): The result of the workflow execution.
                - token (list): A list containing token usage statistics:
                    - Total tokens used.
                    - Prompt tokens used.
                    - Completion tokens used.
        """
        token = []
        workflow = StateGraph(State)
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.retrieval_grade)  # grade documents
        workflow.add_node("rag_generate", self.rag_generate)  # rag_generate
        workflow.add_node("plain_feedback", self.plain_generate)  # plain_feedback
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.route_retrieval,
            {
                "rag_generate": "rag_generate",
                "plain_feedback": "plain_feedback",
            },
        )
        workflow.add_edge("rag_generate", END)
        workflow.add_edge("plain_feedback", END)
        compiled_app = workflow.compile()

        with get_openai_callback() as cb:
            output = compiled_app.invoke({"question": query})
            token.append(cb.total_tokens)
            token.append(cb.prompt_tokens)
            token.append(cb.completion_tokens)
        
        # Log the token usage
        logging.info("Total tokens used: %s", cb.total_tokens)
        logging.info("Prompt tokens used: %s", cb.prompt_tokens)
        logging.info("Completion tokens used: %s", cb.completion_tokens)
        # Log the output
        logging.info("Output: %s", output)
        return output, token
