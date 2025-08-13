
# Prodigy - AI Powered WebEx Bot that acts as a prodigy engineer in organization

Backend integration code for a WebEx Bot that will answer the user queries based on the WebEx space/groups knowledge it has gathered. It uses RAG with LLM for deriving final result for the user query.
The project consists of 2 files:

1. webex_space_crawler.py -> Runs as cron, this will crawl through all the WebEx spaces and fetch all the group messages. The messages are then categorized, correlated and summarized. The categorized summaries are then stored into embeddings.
   
2. bot.py -> Actual bot code that runs on Flask. This code acts as a Webhook integration that will be plugged to the WebEx bot.

# AI Rookie - AI Powered WebEx Bot that chimes into users conversation to suggest its inputs
Backend integration code for a WebEx Bot that will chime into users conversations in a WebEx space. It will understand the conversation happening and chimes into the discussion thread with its ideas or suggestions based on LLM KB.

# Jira Jedi - WebEx Bot for JIRA
Backend integration code for a WebEx Bot that will helps you create tasks, fetch summary, log hours into JIRA for your daily tasks. The bot uses WebEx integration and JIRA python module to integrate with JIRA.

# DPChallenge, Recursion, Graphs, Tries, Algorithms, Arrays, LinkedList
This folder contains my solutions to problems from Algoexpert
