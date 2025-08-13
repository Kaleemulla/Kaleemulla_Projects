import requests
import time
import json
import os
import re
import shutil
import requests.adapters
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
from langchain.docstore.document import Document

os.environ["TOKENIZERS_PARALLELISM"] = "false"

rooms_file = "rooms.json"
rooms_lm_file = "rooms_lm.json"

ack_chat_message = "Sure, I will remember this now"
ack_chat_attch_message = "Sure, I will remember these attachment(s) now"
ack_thread_message = "Sure, I will remember the discussions from this thread now"
ack_thread_attach_message = "Sure, I will remember the discussions along with attachments from this thread now"
error_no_attachment = "No attachment found in this message/thread to learn from"

bot_token = os.environ["BOT_TOKEN"]
user_id = os.environ["USER_ID"]

user_token = os.environ["USER_TOKEN"]

room_id = os.environ["ROOM_ID"]

ciscospark_api = "https://api.ciscospark.com"
api_headers = {"Authorization": f"Bearer {user_token}"}


LOADER_MAPPING = {
        ".csv": (CSVLoader, {}),
        # ".docx": (Docx2txtLoader, {}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".enex": (EverNoteLoader, {}),
        ".epub": (UnstructuredEPubLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".odt": (UnstructuredODTLoader, {}),
        ".pdf": (PyMuPDFLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
        # Add more mappings for other file extensions and loaders as needed
    }


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def initialize_vars(rooms, rooms_lm):
    if os.path.isfile(rooms_file):
        with open(rooms_file, 'r') as file:
            contents = file.read()
            if contents:
                rooms = json.loads(contents)

    if os.path.isfile(rooms_lm_file):
        with open(rooms_lm_file, 'r') as file:
            contents = file.read()
            if contents:
                rooms_lm = json.loads(contents)

    return rooms, rooms_lm


def store_vars(rooms, rooms_lm):
    with open(rooms_file, 'w') as file:
        file.write(json.dumps(rooms, indent=2))

    with open(rooms_lm_file, 'w') as file:
        file.write(json.dumps(rooms_lm, indent=2))

def adding_messages_to_db(db, message):
    if message:
        print(message)
        id = db.add_texts(message)
        print("Adding messages to db")
        return id
    else:
        print("Message blank so nothing to update")

def adding_documents_to_db(db, document):
    if message:
        print("Adding documents to db")
        id = db.add_documents(document)
        return id
    else:
        print("Message blank so nothing to update")

def chunk_messages(message):
    text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=10000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(message)
    return chunks

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

def initialize_embeddings():
    #model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
    return embeddings

def initialize_llm():
    print("Initilize llm")
    llm = ChatOllama(model="llama3:8b", temperature=0.5)
    return llm

def load_vector_db(persist_directory, embeddings):
    print("loading vector db")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db

def rag_retrieve(db):
    print("retrieving RAG")
    rag_retriever = db.as_retriever(search_kwargs={'k': 20})
    return rag_retriever

def get_rag_chain(manner_to_reply_in, rag_retriever, llm):
    rag_template = """Answer the question based on the following context in a {manner_to_reply_in} manner:
    {context}
    Question: {question}
    Answer: 
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = (
            {"context": rag_retriever, "question": RunnablePassthrough(),
             "manner_to_reply_in": lambda x: manner_to_reply_in}
            | rag_prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain

def get_manner_to_reply_in(message, manner_keywords):
    manner_to_reply_in = []
    if isinstance(message,str):
        for manner in manner_keywords:
            if manner in message:
                manner_to_reply_in.append(manner)

    elif isinstance(message,list):
        for manner in manner_keywords:
            if any(manner in text for text in message):
                manner_to_reply_in.append(manner)

    if len(manner_to_reply_in) == 0:
        return "professional"     #Default value of manner of bot
    else:
        return ','.join(manner_to_reply_in)
def find_rooms(rooms, rooms_lm):
    api = f"{ciscospark_api}/v1/rooms"
    res = connect(api, "GET", api_headers)

    if res:
        for item in res["items"]:
            room_id = item["id"]
            if room_id not in rooms:
                rooms[room_id] = item

            if room_id not in rooms_lm or not rooms_lm[room_id]:
                api = f"{ciscospark_api}/v1/messages"
                params = {"roomId": room_id, "max": 2}
                res = connect(api, "GET", api_headers, params)
                if res and "items" in res and len(res["items"]) > 1:
                    rooms_lm[room_id] = res["items"][0]["created"]
                else:
                    rooms_lm[room_id] = str()

    return rooms, rooms_lm


def find_persons(rooms, persons):
    api = f"{ciscospark_api}/v1/memberships"

    for room_id in rooms:
        params = {"roomId": room_id}

        res = connect(api, "GET", api_headers, params)

        if res and "items" in res and len(res["items"]) > 0:
            for item in res["items"]:
                person_id = item["personId"]

                if person_id not in persons:
                    persons[person_id] = item["personDisplayName"]

    return persons


def connect(api, method, headers={}, params={}, payload={}):
    try:
        API_SESSION = requests.Session()
        API_SESSION.verify = False
        API_SESSION.headers = headers

        requests.packages.urllib3.disable_warnings()

        res = {}
        if method.lower() == "get":
            res = API_SESSION.get(api, params=params).json()
        if method.lower() == "post":
            res = API_SESSION.post(api, headers=headers, data=payload).json()

        API_SESSION.close()
        return res

    except Exception as e:
        connect(api, method, headers, params, payload)


def fetch_messages(room_id, rooms_lm, parent_rooms, stored_convo):
    message_items = {}
    single_texts = {}
    convo_texts = {}
    room_last_message = str()
    api_first_message = str()
    api_last_message = str()
    api = f"{ciscospark_api}/v1/messages"
    params = {"roomId": room_id, "max": 100}

    if room_id in rooms_lm:
        room_last_message = rooms_lm[room_id]

    for i in range(3):
        time.sleep(1)
        if api_last_message:
            params.update({"before": api_last_message})

        res = connect(api, "GET", api_headers, params)

        if res and "items" in res and len(res["items"]) > 0:
            print(res)
            for item in res["items"]:
                message_id = item["id"]
                message_created = item["created"]
                if not api_first_message:
                    api_first_message = message_created

                api_last_message = message_created

                if message_created == room_last_message:
                    rooms_lm[room_id] = api_first_message
                    return rooms_lm, parent_rooms, message_items, single_texts, convo_texts, stored_convo

                if item["personId"] == user_id:
                    continue  # Skipping bot sent messages (consider adding to stored_convo if you want to continue chime into same thread)

                if "text" not in item:
                    print(f'Text message is not found for message_id {message_id}, {item}')
                    continue

                message = item["text"]

                if "parentId" in item:
                    if "mentionedPeople" in item and user_id in item["mentionedPeople"]:
                        convo_texts[message_id] = message
                    else:
                        parent_id = item["parentId"]
                        if parent_id in stored_convo:
                            stored_convo[parent_id].append(message)
                        else:
                            stored_convo[parent_id] = [message]

                        if parent_id not in parent_rooms:
                            parent_rooms[parent_id] = room_id
                else:
                    if "mentionedPeople" in item and user_id in item["mentionedPeople"]:
                        single_texts[message_id] = message
                    else:
                        if message_id not in stored_convo:
                            stored_convo[message_id] = [message]  # Potential parent text
                    # else ignore single text with no bot mentions (Not supported for now)

                message_items[message_id] = item
        else:
            break

    rooms_lm[room_id] = api_first_message

    return rooms_lm, parent_rooms, message_items, single_texts, convo_texts, stored_convo


def fetch_thread_chats(room_id, parent_id):
    thread_chats = []
    thread_files = []
    parent_chat = []
    thread_chats_items = {}
    api_last_message = str()
    api = f"{ciscospark_api}/v1/messages/{parent_id}"

    res = connect(api, "GET", api_headers, {})  # Fetch parent message first

    if res:
        message = res["text"]

        parent_chat.append(message)
        thread_chats_items.update({res["id"]: res})

    api = f"{ciscospark_api}/v1/messages"
    params = {"roomId": room_id, "max": 100, "parentId": parent_id}

    for i in range(3):
        time.sleep(1)
        if api_last_message:
            params.update({"beforeMessage": api_last_message})

        res = connect(api, "GET", api_headers, params)

        if res and "items" in res and len(res["items"]) > 0:
            print(res)
            for item in res["items"]:
                message_id = item["id"]
                api_last_message = message_id

                if message_id == parent_id:
                    thread_chats = parent_chat + thread_chats[::-1]
                    return thread_chats, thread_chats_items, thread_files

                if item["personId"] == user_id:
                    continue  # Skipping bot sent messages in thread

                message = item["text"]

                if "files" in item:
                    thread_files += item["files"]

                thread_chats_items[message_id] = item
                thread_chats.append(message)
        else:
            break

    thread_chats = parent_chat + thread_chats[::-1]
    return thread_chats, thread_chats_items, thread_files


def post_message(roomId, message, parentId=None):
    api = f"{ciscospark_api}/v1/messages"

    payload = {
        "roomId": roomId,
        "text": message
    }

    if parentId:
        payload.update({"parentId": parentId})

    res = connect(api, "POST", api_headers, None, payload)


def load_webex_attachments(attachments, d_path):
    if not os.path.isdir(d_path):
        os.makedirs(d_path)

    for attachment in attachments:
        d_file = requests.get(attachment, headers=api_headers)

        disposition = d_file.headers["Content-Disposition"]
        filename_match = pf.search(disposition)
        filename = filename_match.group(1)

        with open(f"{d_path}/{filename}", "wb") as file:
            file.write(d_file.content)

    documents = load_documents(d_path, [])

    shutil.rmtree(d_path, ignore_errors=True)

    return documents



if __name__ == "__main__":
    rooms = {}
    rooms_lm = {}
    parent_rooms = {}
    stored_convo = {}
    persons = {}
    persist_directory = '/Users/kaleemulla/Downloads/webexdb'

    raw_keywords = ["memorize", "remember", "learn", "note"]
    raw_doc_keywords = ["doc", "docs", "document", "documents", "attachment", "attachments", "file", "files"]
    raw_mid_keywords = [" ", " from ", " this ", " these ", " from this ", " from these ", " from attached ", " attached "]
    manner_keywords = ["shy", "obnoxious", "funny", "sarcastic", "sad", "pessimistic", "optimistic"]
    keywords = []
    doc_keywords = []

    for keyword in raw_keywords:
        keywords.append(f"{user_name.split(' ')[0]} {keyword}")
        keywords.append(f"{user_name} {keyword}")

        for doc_keyword in raw_doc_keywords:
            for mid_keyword in raw_mid_keywords:
                doc_keywords.append(f"{user_name.split(' ')[0]} {keyword}{mid_keyword}{doc_keyword}")
                doc_keywords.append(f"{user_name} {keyword}{mid_keyword}{doc_keyword}")

    pk = re.compile('|'.join(map(re.escape, keywords)), re.IGNORECASE)
    pdk = re.compile('|'.join(map(re.escape, doc_keywords)), re.IGNORECASE)
    pf = re.compile('filename="(.*)"')

    #llm = Ollama(model="llama3:8b", temperature=0.5)

    rooms, rooms_lm = initialize_vars(rooms, rooms_lm)

    rooms, rooms_lm = find_rooms(rooms, rooms_lm)
    persons = find_persons(rooms, persons)

    while 1:
        rooms_lm, parent_rooms, message_items, single_texts, convo_texts, stored_convo = fetch_messages(room_id,
                                                                                                         rooms_lm,
                                                                                                         parent_rooms,
                                                                                                         stored_convo)
        print(stored_convo)

        store_vars(rooms, rooms_lm)

        chimed_threads = []
        for parent_id, convo in stored_convo.items():
            if len(convo) > 25:
                thread_chats, thread_chat_items, thread_files = fetch_thread_chats(room_id, parent_id)  # To refresh conversational chats and attachments
                documents = load_webex_attachments(thread_files, f'{room_id}/{parent_id}')

                # Here thead_chats has list of all chats, documents has attachments in thread that can be passed as context for llm chime in

                embeddings = initialize_embeddings()
                db_vector = load_vector_db(persist_directory, embeddings)
                llm = initialize_llm()

                rag_retriever = rag_retrieve(db_vector)

                manner_to_reply_in = get_manner_to_reply_in(convo,manner_keywords)
                print(manner_to_reply_in)
                rag_chain = get_rag_chain(manner_to_reply_in, rag_retriever, llm)
                response = rag_chain.invoke(''.join(convo))

                post_message(parent_rooms[parent_id], response, parent_id)
                chimed_threads.append(parent_id)

        for parent_id in chimed_threads:
            del stored_convo[parent_id]

        for message_id, message in convo_texts.items():
            parent_id = message_items[message_id]["parentId"]
            room_id = message_items[message_id]["roomId"]

            thread_chats, thread_chat_items, thread_files = fetch_thread_chats(room_id, parent_id)

            print(f"{thread_chats}")
            embeddings = initialize_embeddings()
            db_vector = load_vector_db(persist_directory, embeddings)
            llm = initialize_llm()

            if pk.search(message):   # This is when keywords are matched
                has_attachments = False
                if pdk.search(message):   # This is when document keywords are matched
                    if thread_files:
                        has_attachments = True
                        documents = load_webex_attachments(thread_files, f'{room_id}/{message_id}')

                        chunks = chunk_documents(documents)

                        print(f"Split into {len(chunks)} chunks of text")
                        print(chunks)
                        adding_documents_to_db(db_vector, chunks)
                    else:
                        post_message(room_id, error_no_attachment, parent_id)

                thread_chats = thread_chats[:-1]
                print(f"{thread_chats}")
                chunks = chunk_messages(''.join(thread_chats))
                adding_messages_to_db(db_vector, chunks)

                if has_attachments:
                    post_message(room_id, ack_thread_attach_message, parent_id)
                else:
                    post_message(room_id, ack_thread_message, parent_id)

            else:
                rag_retriever = rag_retrieve(db_vector)
                manner_to_reply_in = get_manner_to_reply_in(message,manner_keywords)
                print(manner_to_reply_in)
                rag_chain = get_rag_chain(manner_to_reply_in, rag_retriever, llm)
                response = rag_chain.invoke(message)
                post_message(room_id, response, parent_id)

        for message_id, message in single_texts.items():
            room_id = message_items[message_id]["roomId"]
            embeddings = initialize_embeddings()
            db_vector = load_vector_db(persist_directory, embeddings)
            llm = initialize_llm()

            if pdk.search(message):   # This is when document keywords are matched
                if "files" in message_items[message_id]:
                    attachments = message_items[message_id]["files"]
                    documents = load_webex_attachments(attachments, f'{room_id}/{message_id}')

                    chunks = chunk_documents(documents)

                    print(f"Split into {len(chunks)} chunks of text")
                    print(chunks)
                    adding_documents_to_db(db_vector, chunks)

                    post_message(room_id, ack_chat_attch_message, message_id)
                else:
                    post_message(room_id, error_no_attachment, message_id)

            elif pk.search(message):   # This is when keywords are matched
                message = message.replace(f"{user_name} ", "")
                print(message)
                chunks = chunk_messages(message)
                adding_messages_to_db(db_vector, chunks)

                post_message(message_items[message_id]["roomId"], ack_chat_message, message_id)

            else:
                message = message.replace(f"{user_name} ", "")
                rag_retriever = rag_retrieve(db_vector)
                manner_to_reply_in = get_manner_to_reply_in(message,manner_keywords)
                print(manner_to_reply_in)
                rag_chain = get_rag_chain(manner_to_reply_in, rag_retriever, llm)
                response = rag_chain.invoke(message)
                post_message(room_id, response, message_id)

        time.sleep(1)
