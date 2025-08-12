This project creates a WebEx Bot that will answer the user queries based on the WebEx space/groups knowledge it has gathered. It uses RAG with LLM for deriving final result for the user query.

The project consists of 2 files:

webex_space_crawler.py -> Runs as cron, this will crawl through all the WebEx spaces and fetch all the group messages. The messages are then categorized, correlated and summarized. The categorized summaries are then stored into embeddings.
bot.py -> Actual bot code that runs on Flask. This code acts as a Webhook integration that will be plugged to the WebEx bot.
