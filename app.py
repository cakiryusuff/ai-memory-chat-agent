import os
import time
import logging
import requests
from dataclasses import dataclass, field
import chromadb
import psycopg2
from chromadb.config import Settings
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

ollama_model = OpenAIModel(
    model_name='qwen2.5',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

def check_ollama_connection():
    try:
        response = requests.get('http://localhost:11434')
        if response.status_code == 200:
            logging.info("✅ Ollama server is running.")
            return True
        else:
            logging.warning(f"⚠️ Ollama server responded, but with an unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logging.error("❌ Failed to connect to the Ollama server. Is localhost:11434 running?")
        return False

if check_ollama_connection():
    model = OpenAIModel(
        model_name='llama3.2',
        provider=OpenAIProvider(base_url='http://localhost:11434/v1')
    )
else:
    model = "gpt-4o-mini"
    logging.warning("⚠️ Using fallback model 'gpt-4o-mini' due to Ollama connection failure.")

def check_chroma_connection():
    try:
        chroma_client = chromadb.PersistentClient(path="vectordb", settings=Settings(anonymized_telemetry=False))
        collections = chroma_client.list_collections()
        logging.info("✅ ChromaDB connection successful.")
        logging.info(f"📂 Existing collections: {[col.name for col in collections]}")
        return chroma_client
    except Exception as e:
        logging.error("❌ Failed to connect to ChromaDB:", e)
        return None
    
chroma_client = check_chroma_connection()
if chroma_client:
    collection = chroma_client.get_or_create_collection(name="my_collection")

query_agent = Agent(
    model,
    result_type=list[str],
    system_prompt=(
        "You are a first principle reasoning search query AI agent. "
        "Your list of search queries will be ran on an embedding database of all your conversations "
        "you have ever had with the user. With first principles create a Python list of queries to "
        "search the embeddings database for any data that would be necessary to have access to in "
        "order to correctly respond to prompt. Do not explain anything."
        "example: 'Write an email to my car insurance company and create a pursuasive request for them to lower my monthly rate'"
        "['What is the users name?', 'What is the users current auto insurance provider?']"
    )
)

classify_agent = Agent(
    model,
    result_type=str,
    system_prompt=(
        "You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. "
        "You will only respond True or False. "
        "Determine whether the context contains data that directly relates to the search query. "
        "True = directly related, False = unrelated."
    )
)

result_agent = Agent(
    model,
    result_type=str,
    system_prompt=(
        "You are an AI assistant that remembers all previous conversations with the user. "
        "Use recalled conversations for context if relevant, otherwise ignore them. "
        "Do not mention recalling conversations."
    )
)

def get_connect():
    try:
        return psycopg2.connect(
            host="localhost",
            dbname="demodb",
            user="postgres",
            password=os.getenv("DB_PASSWORD"),
            port="5432"
        )
    except Exception as e:
        logging.error("❌ Database connection error.", e)
        return None

def store_conversation(user_chat: str, gpt_chat: str):
    conn = get_connect()
    if conn is None:
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO chat_conversation (user_chat, gpt_chat)
                VALUES (%s, %s)
            """, (user_chat, gpt_chat))
            conn.commit()
            logging.info("✅ Conversation successfully saved to the database.")
    except Exception as e:
        logging.error("❌ An error occurred while saving to the database.:", e)
    finally:
        conn.close()
        
def add_conversation_to_chroma(user_prompt: str):
    document = f"prompt: '{user_prompt}'"
    doc_id = f"id{str(int(time.time() * 1000))}"
    try:
        collection.add(documents=[document], ids=[doc_id])
        logging.info("✅ Conversation successfully added to ChromaDB.")
    except Exception as e:
        logging.error(f"❌ An error occurred while adding the conversation to ChromaDB: {e}")


@result_agent.system_prompt
def add_content(ctx: RunContext[str]) -> str:
    return f"The similar past conversations is {ctx.deps}"

@dataclass
class Classify(BaseNode):
    matched_contexts: set = field(default_factory=set)
    prompt: str = ""
    
    async def run(self, ctx: GraphRunContext[None, str]) -> End[str]:
        logging.info(f"Classify - Matched Contexts: {self.matched_contexts}")
        result = await result_agent.run(user_prompt=self.prompt, deps=list(self.matched_contexts))
        return Save(result=result.data, prompt=self.prompt)

@dataclass
class Retrieve(BaseNode):
    queries: list[str]
    prompt: str
    
    async def run(self, ctx: GraphRunContext[int]) -> Classify:
        logging.info(f"Retrieve - Queries: {self.queries}")
        matched_contexts = set()

        for query in self.queries:
            search_results = collection.query(query_texts=[query], n_results=2)["documents"][0]

            for result in search_results:
                classify_input = f"SEARCH QUERY: {query}, CONTEXT: {result}"
                check = await classify_agent.run(classify_input)
                if check.data.strip() == "True":
                    matched_contexts.add(result)

        return Classify(matched_contexts=matched_contexts, prompt=self.prompt)

@dataclass
class Increment(BaseNode):
    prompt: str
    
    async def run(self, ctx: GraphRunContext[None, str]) -> Retrieve:
        logging.info(f"Increment - User prompt: {self.prompt}")
        
        queries_response = await query_agent.run(user_prompt=self.prompt)
        logging.info(f"Increment - Queries Response: {queries_response}")
        
        return Retrieve(queries=queries_response.data, prompt=self.prompt)

@dataclass
class Save(BaseNode):
    prompt: str = ""
    result: str = ""
    async def run(self, ctx: GraphRunContext[None, str]) -> End[str]:
        add_conversation_to_chroma(self.prompt)
        store_conversation(self.prompt, self.result)
        return End(self.result)

def delete_from_chroma(action: str, delete_all=False):
    if delete_all:
        log_message = "all conversations"
        get_ids = lambda: collection.get()["ids"]
    else:
        log_message = "last conversation"
        get_ids = lambda: collection.get()["ids"][-1:]

    logging.info(f"This call will delete {log_message}. Are you sure? (y/n)")
    confirm = input("Confirm (y/n): ").strip().lower()
    
    if confirm != "y":
        logging.info(f"User chose not to delete {log_message}.")
        return

    logging.info(f"Deleting {log_message}...")
    try:
        collection.delete(get_ids())
        logging.info(f"✅ {log_message.capitalize()} successfully deleted from ChromaDB.")
    except Exception as e:
        logging.error(f"❌ An error occurred while deleting {log_message} from ChromaDB: {e}")

def list_all_conversations():
    try:
        conversations = collection.get()["documents"]
        logging.info("✅ All conversations retrieved from ChromaDB.")
        for i, conversation in enumerate(conversations):
            print(f"{i + 1}: {conversation}")
    except Exception as e:
        logging.error("❌ An error occurred while retrieving conversations from ChromaDB:", e)

graph = Graph(nodes=[Increment, Retrieve, Classify, Save])

while True:
    user_input = input("User: ")
    
    if user_input.strip().lower() == "/exit":
        logging.info("User chose to exit.")
        break
    elif user_input.strip().lower() == "/del_last":
        delete_from_chroma("last conversation")
        continue
    elif user_input.strip().lower() == "/del_all":
        delete_from_chroma("all conversations", delete_all=True)
        continue
    elif user_input.strip().lower() == "/list_all":
        list_all_conversations()
        continue

    result = graph.run_sync(Increment(prompt=user_input))
    logging.info(f"Result: {result.output}")