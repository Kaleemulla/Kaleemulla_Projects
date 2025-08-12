import requests
import time
import json
import os
import re
import shutil
from datetime import datetime
import requests.adapters
from bs4 import BeautifulSoup
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI

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
# from tqdm import tqdm
from langchain.docstore.document import Document

os.environ["TOKENIZERS_PARALLELISM"] = "false"

rooms_lm_file = "rooms_lm.json"

bot_token = os.environ['BOT_TOKEN']
user_id = os.environ['USER_ID']
user_name = "WebEx Space Crawler"

user_token = os.environ['USER_TOKEN']
room_id = os.environ['ROOM_ID']

app_key = os.environ['APP_KEY']
access_token = os.environ['ACCESS_TOKEN']

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


def initialize_vars(rooms_lm):
    if os.path.isfile(rooms_lm_file):
        with open(rooms_lm_file, 'r') as file:
            contents = file.read()
            if contents:
                rooms_lm = json.loads(contents)

    return rooms_lm


def store_vars(rooms_lm):
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
    text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=500, chunk_overlap=20,
                                                   length_function=len)
    chunks = text_splitter.split_text(message)
    return chunks


def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks


def initialize_embeddings():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model


def load_vector_db(persist_directory, embeddings):
    print("loading vector db")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db


def find_rooms(rooms_lm):
    api = f"{ciscospark_api}/v1/rooms"
    res = connect(api, "GET", api_headers)

    if res:
        for item in res["items"]:
            room_id = item["id"]

            if room_id not in rooms_lm:
                rooms_lm[room_id] = str()

    return rooms_lm


def find_members(room_id, members):
    api = f"{ciscospark_api}/v1/memberships"

    params = {"roomId": room_id}

    res = connect(api, "GET", api_headers, params)

    if res and "items" in res and len(res["items"]) > 0:
        for item in res["items"]:
            person_id = item["personId"]

            if person_id not in members:
                members[person_id] = item["personDisplayName"]

    return members


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
        print(e)
        connect(api, method, headers, params, payload)


def fetch_messages(room_id, rooms_lm, members):
    global api_headers

    api = f"{ciscospark_api}/v1/messages"
    params = {"roomId": room_id, "max": 100}

    processed_until_msg = str()
    new_processed_until_msg = str()

    if room_id in rooms_lm:
        processed_until_msg = rooms_lm[room_id]

    loop = True
    single_messages = {}
    thread_messages = {}
    incomplete_threads = set()

    while loop:
        res = connect(api, "GET", api_headers, params)
        # print(res)

        if not res["items"]:
            rooms_lm[room_id] = new_processed_until_msg
            break

        for item in res["items"]:
            message_id = item["id"]
            iso_timestamp = item["created"]
            dt = datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            message_date = dt.date()

            if not new_processed_until_msg:
                new_processed_until_msg = message_id

            if message_id == processed_until_msg:
                rooms_lm[room_id] = new_processed_until_msg
                loop = False
                break

            if "text" not in item:
                print(f'Text message is not found for message_id {message_id}, {item}')
                continue

            message = item["text"]
            person_id = item["personId"]

            if person_id in members:
                person = members[person_id]
            else:
                person = item["personEmail"].split('@')[0]

            if "parentId" in item:
                parent_id = item["parentId"]
                message = f'{message_date}: {person}: {message}'
                incomplete_threads.add(parent_id)   # Lets assume it is incomplete thread chat

                if parent_id in thread_messages:
                    thread_messages[parent_id].append(message)
                else:
                    thread_messages[parent_id] = [message]
            else:
                if "html" in item and "</blockquote>" in item["html"]:  # Parsing the quoted message replies
                    html = item["html"].split("</blockquote>")[1]
                    soup = BeautifulSoup(html, 'html.parser')
                    texts = []
                    for tag in soup.find_all():
                        if not tag.find():  # Only leaf nodes
                            text = tag.get_text()
                            if tag.name == "code" and tag.get("class"):
                                # Add extra newline for <code class="...">
                                texts.append('\n'+text)
                            else:
                                texts.append(text)

                    if not texts:
                        texts.append(html)

                    reply_text = "\n".join(texts)
                    quote_text = str()

                    if reply_text:
                        quote_text = message.split(reply_text)[0]
                        reply_text = f'{message_date}: {person}: {reply_text}'

                    quote_text = re.sub('\n', ': ', quote_text, 1)

                    # print([quote_text], [reply_text])

                    if message_id in thread_messages:
                        thread_messages[message_id].append(f'{quote_text}{reply_text}')
                        if message_id in incomplete_threads:
                            incomplete_threads.remove(message_id)   # Remove this message_id as parent chat message is found
                    else:
                        single_messages[message_id] = f'{quote_text}{reply_text}'

                else:
                    message = f'{message_date}: {person}: {message}'
                    if message_id in thread_messages:
                        thread_messages[message_id].append(message)
                        if message_id in incomplete_threads:
                            incomplete_threads.remove(message_id)   # Remove this message_id as parent chat message is found
                    else:
                        single_messages[message_id] = message

        params.update({"beforeMessage": res["items"][-1]["id"]})    # current processing until message

    return rooms_lm, single_messages, thread_messages, incomplete_threads


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
    members = {}
    persist_root_directory = '/Users/kasharie/Downloads/spacedb'
    # persist_root_directory = '/home/cmsbuild/kaleem/spacedb'

    rooms_lm = initialize_vars(rooms_lm)

    rooms_lm = find_rooms(rooms_lm)

    openai_client = AzureOpenAI(azure_endpoint='https://chat-ai.cisco.com',
                         api_key=access_token,
                         api_version="2024-12-01-preview")

    while 1:
        members = find_members(room_id, members)
        rooms_lm, single_messages, thread_messages, incomplete_threads = fetch_messages(room_id, rooms_lm, members)

        # {"id1": message-1, "id2": message-2, "id3": message3, ..}
        print(f'Single Messages: {list(single_messages.values())}')

        # {"parentid1": [thread-message-1, thread-message-2], "parentid2": [thread-message-1, ..], ..}
        print(f'Thread Messages: {list(thread_messages.values())}')

        thread_summaries = []
        thread_ids = []
        for pid, chat_thread in thread_messages.items():
            print(f'Creating thread messages summary for item {pid}')
            chat_thread_text = '\n'.join(chat_thread[::-1])
            chat_thread_prompt = f"""
                From below chat messages, group the relevant ones and summarize with minute details. Return only the summary 
                {chat_thread_text}
            """

            message = [
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": chat_thread_prompt}]

            response = openai_client.chat.completions.create(model="gpt-4o",
                                                      messages=message,
                                                      user=f'{{"appkey": "{app_key}"}}')

            if response:
                thread_summaries.append(response.choices[0].message.content)
                thread_ids.append(pid)

        singles_summaries = []
        if single_messages:
            all_single_messages = list(single_messages.values())
            for i in range(0, len(all_single_messages), 50):
                print(f'Creating single messages summary for items {i} to {i+50}')
                chat_singles_text = '\n'.join(all_single_messages[i:i+50])
                chat_singles_prompt = f"""
                    From below chat messages, group the relevant ones and summarize with minute details. Return only the summary 
                    {chat_singles_text}
                """

                message = [
                    {"role": "system", "content": "You are a chatbot"},
                    {"role": "user", "content": chat_singles_prompt}]

                response = openai_client.chat.completions.create(model="gpt-4o",
                                                          messages=message,
                                                          user=f'{{"appkey": "{app_key}"}}')

                if response:
                    singles_summaries.append(response.choices[0].message.content)

        store_vars(rooms_lm)

        db_client = chromadb.PersistentClient(path=f'{persist_root_directory}/{room_id}')
        collection = db_client.get_or_create_collection(name="space_messages")
        embedding_model = initialize_embeddings()

        vectorstore = Chroma(
            client=db_client,
            collection_name="space_messages",
            embedding_function=embedding_model,
        )

        if thread_summaries:
            print("Adding thread_summaries")
            vectorstore.add_texts(texts=thread_summaries, ids=thread_ids)

        if singles_summaries:
            print("Adding singles_summaries")
            vectorstore.add_texts(singles_summaries)

        time.sleep(2)
