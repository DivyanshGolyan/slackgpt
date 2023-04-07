from slack_sdk.web.async_client import AsyncWebClient
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from tenacity import retry, stop_after_attempt, wait_random_exponential
import asyncio
import openai
import PyPDF2
import aiohttp
import tiktoken
import re
import json
import sqlite3
import aiosqlite
import datetime
from threading import local
import time
import io
import os

SLACK_BOT_TOKEN = ""
SLACK_APP_TOKEN = ""
OPENAI_API_KEY = ""

# Initialize the Slack bot and API clients
app = AsyncApp(token=SLACK_BOT_TOKEN)
client = AsyncWebClient(SLACK_BOT_TOKEN)


# Create a table to store the conversation history
async def initialize_database():
    async with aiosqlite.connect("data.db") as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                channel_id  TEXT NOT NULL,
                type TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                conversation TEXT NOT NULL
            )
            """)

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                message_ts TEXT NOT NULL,
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                message_text TEXT NOT NULL,
                token_count INTEGER
            )
            """)


async def execute_query(query, *params):
    async with aiosqlite.connect("data.db") as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, *params)
            await conn.commit()


async def download_file(url: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            file = io.BytesIO(await response.read())
            return file


async def extract_text_from_txt(txt_content: io.BytesIO) -> str:
    text = txt_content.getvalue().decode("utf-8")
    return text


async def extract_text_from_pdf(pdf_content: io.BytesIO) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_content)
        text = ""

        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()

        return text

    except PyPDF2.errors.PdfReadError:
        return "Error: Could not read the PDF file. It might be corrupted or not properly formatted."


def num_tokens_from_messages(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print(
            "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print(
            "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 4
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


async def get_user_name(user):
    user_info = await app.client.users_info(user=user)
    user_name = user_info["user"]["name"]

    return user_name


@app.error
async def global_error_handler(error, body, logger):
    logger.exception(f"Error in the app: {error}")
    print(f"Error in the app: {error}")
    print(f"Error body: {body}")


@app.shortcut("summarise_thread")
async def summarise_thread(ack, body, logger, client):
    # Acknowledge the shortcut request
    await ack()

    print(body)
    # Get the thread_ts and channel_id from the message
    event_ts = body["message"]["ts"]
    is_threaded = "thread_ts" in body["message"]
    thread_ts = body["message"]["thread_ts"] if is_threaded else event_ts
    user = body["user"]["id"]

    #    user_info = await app.client.users_info(user=user)
    user_name = await get_user_name(user)
    channel_id = body["channel"]["id"]

    #    await app.client.chat_delete(channel=channel_id, ts=event_ts)

    try:
        await app.client.conversations_join(channel=channel_id)
    except Exception as e:
        if "channel_not_found" in str(e):
            await client.chat_postMessage(
                channel=user,
                text=
                f"Please add me to a private channel (channel name –> Integrations –> Add an app –> Search for and add ChatGPT) before using the message shortcut."
            )
        else:
            print(f"Error occurred while joining the conversation: {e}")
            pass

    # Get all the messages in the thread
    result = await client.conversations_replies(channel=channel_id,
                                                ts=thread_ts)
    thread_messages = result["messages"]

    # Collate all the messages in the thread
    collated_text = ""
    for message in thread_messages:
        collated_text += f"User {await get_user_name(message['user'])}:\n {message['text']}\n\n"

    # Regex pattern to search for user_ids enclosed in <>
    pattern = re.compile(r"<@(U[0-9A-Z]+)>")

    async def replace_user_ids_async(collated_text, pattern):
        modified_text = collated_text
        matches = list(pattern.finditer(collated_text))

        for match in matches:
            user_id = match.group(1)
            user_name = await get_user_name(user_id)
            modified_text = modified_text.replace(match.group(0),
                                                  f"@{user_name}")

        return modified_text

    modified_text = await replace_user_ids_async(collated_text, pattern)

    # print(modified_text)

    conversation = [{
        "role":
        "system",
        "content":
        f"You are a helpful assistant. Summarise key comments from the below message thread for me. The conversation is happening on Slack. Ensure the content is clear and engaging. Present facts objectively. Ensure that the overall sentiment expressed in the messages is accurately reflected. Optimize for highly original content. Ensure its written professionally, in a way that is appropriate for the situation. I am {user_name}. If you refer to users in the summary, please use their first names only; except if the you need to refer to me, use the second-person point of view."
    }]

    await execute_query(
        "INSERT INTO conversation_history (user_id, channel_id, type, thread_ts, conversation) VALUES (?, ?, ?, ?, ?)",
        (user, channel_id, "summarise thread", thread_ts,
         json.dumps(conversation)))

    # Append the user message to the conversation history
    conversation.append({"role": "user", "content": modified_text})
    
    await execute_query(
        "UPDATE conversation_history SET conversation = ? WHERE thread_ts = ? and type = ? and user_id = ? and channel_id = ?",
        (json.dumps(conversation), thread_ts, "summarise thread", user,
         channel_id))

    # Get the current Indian Standard Time for message timestamp
    ist = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    message_ts = ist.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Get the prompt token count
    promptTokenCount = await asyncio.to_thread(num_tokens_from_messages,
                                               conversation, "gpt-4")
    print(f"Token count: {promptTokenCount}")

    # Insert the user's message into the messages table
    await execute_query(
        """
    INSERT INTO messages (type, thread_ts, message_ts, user_id, channel_id, sender, message_text, token_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, ("summarise thread", thread_ts, message_ts, user, channel_id, "User",
          modified_text, promptTokenCount))

    # Send the collated string to the OpenAI API
    # Set OpenAI API key and call the API with the conversation history
    openai.api_key = OPENAI_API_KEY
    assistant_response, completion_tokens = await asyncio.to_thread(
        call_openai_api, conversation)

    # Send the assistant's response to the user
    if is_threaded:
        await client.chat_postEphemeral(
            channel=channel_id,
            text=f"{assistant_response}",
            thread_ts=thread_ts,
            user=user  # Send the response in the same thread
        )
    else:
        await client.chat_postMessage(
            channel=user,
            text=f"Requested Message Summary:\n{assistant_response}")

    # Get the current Indian Standard Time for message timestamp
    ist = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    message_ts = ist.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Insert the assistant's response into the messages table
    await execute_query(
        """
        INSERT INTO messages (type, thread_ts, message_ts, user_id, channel_id, sender, message_text, token_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, ("summarise thread", thread_ts, message_ts, user, channel_id,
          "Assistant", assistant_response, completion_tokens))


@app.event("message")
async def handle_message_events(body, logger):

    # Ignore non-direct messages
    if body["event"].get("channel_type") != "im":
        return

    # Extract relevant information from the event
    channel_id = body["event"]["channel"]
    event_ts = body["event"]["event_ts"]
    user = body["event"]["user"]
    user_message = str(body["event"]["text"])

    # Check if the message is part of a thread
    is_threaded = "thread_ts" in body["event"]
    thread_ts = body["event"]["thread_ts"] if is_threaded else event_ts

    # print(channel_id)
    # print(thread_ts)
    # print(event_ts)
    print("Received event:", body)

    # If message contains files, extract text from them
    extracted_text = ""
    if "files" in body["event"]:
        for file in body["event"]["files"]:
            url = file["url_private"]
            file_content = await download_file(url, SLACK_BOT_TOKEN)

            # Check if the file is a Slack snippet or text file
            if file["mode"] == "snippet" or file["mode"] == "post" or file[
                    "filetype"] == "text":
                extracted_text += f"From {file['name']}:\n{await extract_text_from_txt(file_content)}\n"
            elif file["filetype"] in ["pdf"]:
                extracted_text += f"From {file['name']}:\n{await extract_text_from_pdf(file_content)}\n"
            else:
                error_message = f"Unsupported file type: {file['filetype']}. Please send PDF or text files only."
                await client.chat_postMessage(channel=channel_id,
                                              text=error_message)
                return

    # Combine user_message and extracted_text
    combined_message = (
        user_message + "\n" +
        extracted_text.strip()) if "files" in body["event"] else user_message

    # Log message
    #    print(f"{channel_id}: {combined_message}")

    # Check if the conversation history exists in the SQLite table
    async with aiosqlite.connect("data.db") as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT conversation FROM conversation_history WHERE thread_ts = ? and type = ?",
                (thread_ts, "direct message"))
            conversation = await cursor.fetchone()

    # If the conversation history doesn't exist, create it
    if conversation is None:
        conversation = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]
        await execute_query(
            "INSERT INTO conversation_history (user_id, channel_id, type, thread_ts, conversation) VALUES (?, ?, ?, ?, ?)",
            (user, channel_id, "direct message", thread_ts,
             json.dumps(conversation)))
    else:
        conversation = json.loads(conversation[0])

    # Append the user message to the conversation history
    conversation.append({"role": "user", "content": combined_message})
    await execute_query(
        "UPDATE conversation_history SET conversation = ? WHERE thread_ts = ? and type = ? and user_id = ? and channel_id = ?",
        (json.dumps(conversation), thread_ts, "direct message", user,
         channel_id))

    # Get the current Indian Standard Time for message timestamp
    ist = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    message_ts = ist.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Get the prompt token count
    promptTokenCount = await asyncio.to_thread(num_tokens_from_messages,
                                               conversation, "gpt-4")
    print(f"Token count: {promptTokenCount}")

    # Insert the user's message into the messages table
    await execute_query(
        """
    INSERT INTO messages (type, thread_ts, message_ts, user_id, channel_id, sender, message_text, token_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, ("direct message", thread_ts, message_ts, user, channel_id, "User",
          combined_message, promptTokenCount))

    # Set OpenAI API key and call the API with the conversation history
    openai.api_key = OPENAI_API_KEY
    assistant_response, completion_tokens = await asyncio.to_thread(
        call_openai_api, conversation)

    # Extract the assistant's response and update the conversation history in SQLite
    conversation.append({"role": "assistant", "content": assistant_response})
    await execute_query(
        "UPDATE conversation_history SET conversation = ? WHERE thread_ts = ? and type = ? and user_id = ? and channel_id = ?",
        (json.dumps(conversation), thread_ts, "direct message", user,
         channel_id))

    # Send the assistant's response to the user
    await client.chat_postMessage(
        channel=channel_id,
        text=f"{assistant_response}",
        thread_ts=thread_ts,  # Send the response in the same thread
    )

    # Get the current Indian Standard Time for message timestamp
    ist = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    message_ts = ist.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Insert the assistant's response into the messages table
    await execute_query(
        """
        INSERT INTO messages (type, thread_ts, message_ts, user_id, channel_id, sender, message_text, token_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, ("direct message", thread_ts, message_ts, user, channel_id,
          "Assistant", assistant_response, completion_tokens))


# Define a function to call the OpenAI API with retry logic
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai_api(messages, model='gpt-4'):
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages)
        print(response)
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        message_content = choice.message["content"]
        completion_tokens = response.usage["completion_tokens"]

        if finish_reason == "length":
            return "Error: The response was cut off due to exceeding the maximum token limit. Please try again with a shorter conversation.", None
        else:
            return message_content, completion_tokens

    except openai.error.InvalidRequestError as e:
        error_message = str(e)
        if "maximum context length" in error_message:
            return "Error: The conversation is too long to process. Please try shortening the conversation or removing less relevant parts.", None
        elif "That model is currently overloaded with other requests" in error_message and model != 'gpt-3.5-turbo':
            return call_openai_api(messages, model='gpt-3.5-turbo')
        else:
            return f"Error: {e}", None
    except Exception as e:
        return f"Error: {e}", None


# Start the Slack bot


async def main():
    await initialize_database()
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
