# 3 agents work together:
#   1. Search in RAD Database
#   2. Search in Web
#   3. Update in file
# import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from crewai.tools import tool

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = LLM(model="gpt-4o-mini")

search_tool = SerperDevTool()

# 1. Build RAG
def build_vector_db():
    
    loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

vector_db = build_vector_db()

# 2. Create RAG Tool
@tool("Customer Support Knowledge Base")
def customer_support_rag_tool(query: str) -> str:
    """
    Use this tool to search company FAQs, policies, pricing, refund rules,
    product documentation, and customer support knowledge base.
    """
    results = vector_db.similarity_search(query, k=4)
    context = "\n\n".join([ doc.page_content for doc in results ])
    return context

@tool("Save to File")
def save_to_file(content: str) -> str:
    """
    Saves the provided text content (query, answer 1, and answer 2) 
    into a text file named support_log.txt.
    """
    with open("support_log.txt", "a", encoding="utf-8") as f:
        f.write(content + "\n" + "="*50 + "\n")
    return "Successfully saved to support_log.txt"

# 3. Create Customer Support Agent
# ----- Agent 1: find a response from rad -----
support_agent = Agent(
    role="Customer Support Agent",
    goal="Answer customer queries accurately using the knowledge base.",
    backstory=(
        "You are a helpful customer support agent. "
        "You must answer only using the provided knowledge base. "
        "If the answer is not available, politely say that you don't have enough information "
        "and suggest contacting the support team."
    ),
    tools=[customer_support_rag_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ----- Agent 2: Web search -----
websearch_agent = Agent(
    role="Web Researcher",
    goal="Search the web and find the relevant content from topic",
    backstory="You are a web researcher for years.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

# ----- Agent 3: do SEO on the blog -----
log_agent = Agent(
    role="Log update agent",
    goal="Compile the conversation history, current query, Assistant's answer, and Web Search answer. Write them to a text file, and return both answers clearly formatted.",
    backstory="You are a meticulous records keeper ensuring every query and its associated answers are logged safely in a text file and returned cleanly to the UI.",
    tools=[save_to_file],
    verbose=True,
    allow_delegation=False,
    llm=llm
)


# 4. Run Query
def answer_customer(query: str):
    task1 = Task(
        description=f"""
        Customer asked:

        {query}

        Search the knowledge base using the RAG tool.
        Give a clear, friendly, and professional customer support answer.
        Do not hallucinate.
        """,
        expected_output="A helpful customer support response based only on the knowledge base.",
        agent=support_agent
    )
    task2 = Task(
        description=
        f"""Perform web search and provide response for  {query} \n\n"
        "Using the current query and context, search the web and write an answer based strictly on current web search results."
        """,
        expected_output="An answer to the query derived from web search results.",
        agent=websearch_agent
    )

    task3 = Task(
        description=(
            f"1. Take current query: '{query}'.\n"
            "2. Take the output from Task 1 (Assistant's Answer).\n"
            "3. Take the output from Task 2 (Web Search Assistant's Answer).\n"
            "4. Format them nicely and use your 'Save to File Tool' to write all data to 'support_log.txt'.\n"
            "6. Finally, output a clean markdown string that displays BOTH the Assistant's Answer and the Web Search Answer for the user."
        ),
        expected_output="A formatted markdown string containing both Answer 1 and Answer 2.",
        agent=log_agent,
        context=[task1, task2]
    )

    crew = Crew(
        agents=[support_agent, websearch_agent, log_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return result

#================================================
# Trigger
#================================================
if __name__ == "__main__":
    query = input("Customer Query: ")
    response = answer_customer(query)
    print("\nAgent Response:\n")
    print(response)