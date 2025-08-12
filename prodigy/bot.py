import requests
import json
import os
import logging
from logging.handlers import RotatingFileHandler
import requests.adapters
from openai import AzureOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import SentenceTransformer

from flask import Flask
from flask import request

os.environ["TOKENIZERS_PARALLELISM"] = "false"

rooms_lm_file = "rooms_lm.json"

user_token = os.environ['USER_TOKEN']

bot_email = "team-prodigy@webex.bot"
bot_name = "Prodigy"
bot_id = os.environ['BOT_ID']
bot_token = os.environ['BOT_TOKEN']

room_id = os.environ['ROOM_ID']

app_key = os.environ['APP_KEY']
access_token = str()

token_url = "https://id.cisco.com/oauth2/default/v1/token"

ciscospark_api = "https://api.ciscospark.com"
api_headers = {"Authorization": f"Bearer {user_token}"}

app = Flask(__name__)

def initialize_embeddings():
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


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
        #print(e)
        connect(api, method, headers, params, payload)


def send_get(url, payload=None, js=True):
    if payload == None:
        request = requests.get(url, headers=headers)
    else:
        request = requests.get(url, headers=headers, params=payload)
    if js == True:
        request = request.json()
    return request


def send_post(url, data):
    request = requests.post(url, json.dumps(data), headers=headers).json()
    return request


def post_message(room_id, message, parentId=None):
    logger.debug("Posting message to " + room_id)
    logger.debug(message)

    payload = {"roomId": room_id, "html": message}

    if parentId:
        payload.update({"parentId": parentId})

    return requests.post(f"{ciscospark_api}/v1/messages", json.dumps(payload), headers=headers).json()


def post_message_markdown(room_id, message):
    logger.debug("Posting message to " + room_id)
    logger.debug(message)
    payload = {"roomId": room_id, "markdown": message}

    return requests.post(f"{ciscospark_api}/v1/messages", json.dumps(payload), headers=headers).json()


def init_logger():
    logger = logging.getLogger("botlog")
    logging.basicConfig(
        handlers=[
            RotatingFileHandler("prodigy.log", maxBytes=10000, backupCount=10)],
        level="DEBUG",
        format='%(asctime)s %(levelname)-8s %(thread)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger.debug("Initializing logger")
    return logger


@app.route('/', methods=['GET', 'POST'])
def prodigy_webhook():
    print(request)
    if request.method == 'GET':
        return "Success"

    webhook = request.get_json(silent=True)
    room_id = webhook["data"]["roomId"]
    print(webhook)

    if webhook["resource"] == "room" and webhook["event"] == "created":
        post_message_markdown(room_id,
                              "<@all><BR>Welcome to a new room with Prodigy!! You have to call me specifically with `@%s` to get response." % bot_name)
        return "true"

    if webhook["resource"] == "memberships" and webhook["data"]["roomType"] == "group":
        post_message_markdown(room_id,
                              "<@all><BR>Hello from Prodigy!! As this is a group room, you have to call me specifically with `@%s` to get response." % bot_name)
        return "true"

    email = webhook["data"]["personEmail"]

    if webhook["resource"] == "memberships":
        if email == bot_email:
            msg = "Hi there, you can post me your queries."
        else:
            msg = "Welcome <@personEmail:{}> to the group room. If you want me to answer, you need to call me specifically with @<@personId:{}>".format(
                email, bot_id)

        post_message_markdown(room_id, msg)
        return "true"

    if "@webex.bot" not in email:
        try:
            message_id = webhook['data']['id']
            member_id = webhook['data']['personEmail']
            message_created = webhook['data']['created']
            jira_user = webhook['data']['personEmail']

            result = send_get(f'{ciscospark_api}/v1/messages/{message_id}')
            query = result.get('text', '')
            print(f'Raw message: {query}')

            for name in bot_name.split(" "):
                query = query.replace(name + " ", '')
            query = query.replace(bot_name + " ", '')

            print(f'Processing Query: {query}')

            auth_key = os.environ['AUTH_KEY']

            token_headers = {
                "Accept": "*/*",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {auth_key}",
            }
            token_payload = "grant_type=client_credentials"

            token_response = connect(token_url, "POST", token_headers, {}, token_payload)
            access_token = token_response.get("access_token")

            persist_directory = "/opt/spacedb"

            client = chromadb.PersistentClient(path=f'{persist_directory}/{room_id}')
            collection = client.get_or_create_collection(name="space_messages")
            embedding_model = initialize_embeddings()

            vectorstore = Chroma(
                client=client,
                collection_name="space_messages",
                embedding_function=embedding_model,
            )

            model = SentenceTransformer("all-MiniLM-L6-v2")

            query_embedding = model.encode([query])[0].tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )

            retrieved_context = "\n".join(results["documents"][0])

            client = AzureOpenAI(azure_endpoint='https://chat-ai.cisco.com',
                                 api_key=access_token,
                                 api_version="2024-12-01-preview")

            rag_prompt = f"""
                You are an AI assistant helping answer questions based on internal group chat messages.
                
                Here are some chat messages:

                {retrieved_context}
                
                Based on this information, answer the following question in a short summary. IFF you cant find answer from given context then answer based on your own knowledge:

                Question: {query}
                Answer:
                """

            response = client.chat.completions.create(
                model="gpt-4o",
                user=f'{{"appkey": "{app_key}"}}',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": rag_prompt},
                ],
                temperature=0.1
            )

            send_message = response.choices[0].message.content
            post_message(room_id, send_message)
            return '', 200

        except Exception as e:
                print(e)
                logger.error(e)
                return '', 500


if __name__ == "__main__":
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": "Bearer " + bot_token
    }

    logger = init_logger()
    ip = "0.0.0.0"
    port = 8001


    '''if bot_token:
            test_auth = send_get(f"{ciscospark_api}/v1/people/me", js=False)
            if test_auth.status_code == 401:
                print("Looks like the provided access token is not correct.\n")
                sys.exit()
            if test_auth.status_code == 200:
                test_auth = test_auth.json()
                bot_name = test_auth.get("displayName", "")
                bot_email = test_auth.get("emails", "")[0]
    
                if "@webex.bot" not in bot_email:
                    logger.error("You have provided an access token which does not relate to a Bot Account.")
                    print("You have provided an access token which does not relate to a Bot Account.\n")
                    sys.exit()
        else:
            print("'bearer' variable is empty! \n")
            sys.exit()'''

    
    app.run(host=ip, port=port, ssl_context=('/home/cmsbuild/jira_jedi/cert.pem', '/home/cmsbuild/jira_jedi/key.pem'))
