from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOpenAI(model="gpt-4o", model_kwargs={"response_format": {"type": "json_object"}})
parser = StrOutputParser()

opening_prompt_template = '''
        You will return a JSON with one key "answer" and a second one called "compressed_answer" which is aggressively compressed and summarized in a way that you can read it but it doesn't need to be human-readable. Let's play a text-based RPG. First we'll set the stage for my adventure. You will write part of the story and at the end of your response you will give 3 choices to continue the story. The 4th option will be to regenerate what you just wrote. You will also leave a 5th option open for the user to tell you how to drive the story. We'll first crate the mood and setting if the story: 

        Setting/mood (e.g. sci-fi, cute animals): {story_setting}

        Now we'll create my character:

        1. Name: {character_name}
        2. Character physical description: {character_physical_description}
        3. Character description: {character_psychological_description}
        4. Character background story: {character_story}

        And these are other characters and sidekicks:

        Sidekicks: {sidekicks}
        Storyline: {storyline}
'''

opening_prompt_w_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert RPG story teller."),
        ("user", opening_prompt_template)]
)

opening_prompt_data = {
    "story_setting": "sci-fi in Mexico including places like Tepito, Polanco San Cristobal de las Casas and Palenque",
    "character_name": "Javier",
    "character_physical_description": "Short, bulky, wears a high tech suit that looks like indigenous clothing",
    "character_psychological_description": "He gets easily triggered into violent tantrums because of past trauma that we must uncover",
    "character_story": "He was a former soldier who was part of a secret government project that went wrong",
    "sidekicks": "John, an American dissident and former cyborg-race runner. Maria, an old indigenous woman with a long history of mental illness and psychic spiritual powers",
    "storyline": ""
    }

chain = opening_prompt_w_template | chat | parser
result = chain.invoke(opening_prompt_data)
print(result)

# How to continue a conversation instead of a doing single completion
# * The problem with large conversations is that token usage escalates rapidly. Possible solutions include:
# * Use Assistans API V2 by uploading the messages as a text file in as embeddings/chunked format.
# * Discard older conversation turns that won’t fit into the remaining context space after calculating tokens locally for prompt, user input, max_tokens reservation.
# * Use last N interactions
# * Use another AI to periodically summarize turns or the oldest conversation.
# * Use a vector database and AI embeddings to remember the whole conversation and pass prior exchanges that are most relevant to the current conversation flow.
    # TODO 
    # * Locally: https://pypi.org/project/vectordb/