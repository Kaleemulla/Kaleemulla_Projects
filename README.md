
# Prodigy - AI Powered WebEx Bot that acts as a prodigy engineer in organization

AI powered WebEx Bot that answers user queries based on the knowledge it has gathered by crawling all the webex groups. The group's discussions are analyzed, correlated and summarised to create embeddings that are used as RAG with LLM to generate the results for the user query.

1. webex_space_crawler.py -> Runs as cron, this will crawl through all the WebEx spaces and fetch all the group messages. The messages are then categorized, correlated and summarized. The categorized summaries are then stored into embeddings.
   
2. bot.py -> Actual bot code that runs on Flask. This code acts as a Webhook integration that will be plugged to the WebEx bot.

# AI Rookie - AI Powered WebEx Bot that chimes into users conversation to suggest its inputs
AI Powered WebEx Bot that chime's into users conversations in a WebEx space. It will understand the conversation happening and chime's into the discussion thread with its own ideas or suggestions based on LLM KB + RAG. It can also be actively trained to remember the informations on WebEx space like chat messages/threads, documents attached in space etc. The bot generates embeddings that are then used as RAG with LLM to generate the response.

# Jira Jedi - WebEx Bot for JIRA
Backend integration code for a WebEx Bot that will helps you create tasks, fetch summary, log hours into JIRA for your daily tasks. The bot uses WebEx integration and JIRA python module to integrate with JIRA.

# DPChallenge, Recursion, Graphs, Tries, Algorithms, Arrays, LinkedList
This folder contains my solutions to problems from Algoexpert
