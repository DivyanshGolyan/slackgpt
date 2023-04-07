# Slack ChatGPT Project
## Table of Contents
1. [Introduction](#introduction)
2. [Technologies](#technologies)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Features](#features)
## Introduction
This Slack bot is designed to be a helpful assistant for summarizing message threads and answering direct messages within Slack. It uses OpenAI's GPT-4 API to generate human-like text summaries and responses based on the provided conversation.
## Technologies
The Slack ChatGPT Project is built using:
* Python 3
* openai 0.27.2
* slack-sdk 3.20.2
* slack-bolt 1.16.4
* PyPDF2 3.0.1
* aiohttp 3.8.4
* aiosqlite 0.18.0
* tiktoken 0.3.3
* tenacity 8.2.2

## Setup
1. Clone the project from the GitHub repository.
2. Install the required Python packages.
   
   pip install -r requirements.txt
   
3. Create a slack app. Use the app manifest in manifest.json.
4. Set your OpenAI API key and Slack tokens as environment variables.
   
   export OPENAI_API_KEY="your-openai-api-key"
   export SLACK_BOT_TOKEN="your-slack-bot-token"
   export SLACK_APP_TOKEN="your-slack-app-token"
   
## Usage
1. Run the bot using the following command.
   
   python app.py
   
2. Within Slack, users can interact with the Slack bot in two ways:
   * Summarizing message threads using the "summarise_thread" shortcut on the message's action menu.
   * Engaging in one-on-one direct messages with the bot.
## Features
* **Thread Summarization**: The bot summarizes the key comments from a message thread, capturing the overall sentiment expressed in the messages.
* **Direct Messaging**: Users can engage in direct messaging conversations with the bot, and the bot will respond based on the conversation history.
* **Automatic User Mention Conversion**: The bot converts user mentions (@user) into user names or second-person point of view for the user interacting with the bot.
* **File Support**: The bot can process and extract the text from text files, PDF files, and Slack snippets to include in its responses.
* **Customizable Model**: Users can easily provide a different OpenAI model by modifying the call_openai_api function.
