import os
import backoff
import openai
from pprint import pprint
from dotenv import load_dotenv
from openai import OpenAIError

# Color helper class.
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class LLMModelWrapper:
    def __init__(
        self,
        system_message,
        model="gpt-3.5-turbo",
        # temperature=0.6,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        self._model = model
        self._messages = [{"role": "system", "content": system_message}]
        # self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=60)
    def _chat_completion(self):
        return openai.chat.completions.create(
            model=self._model,
            messages=self._messages,
            # temperature=self._temperature,
            top_p=self._top_p,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
        )

    def step(self, user_input, verbose, agent_name=""):
        if verbose >= 2:
            print(f"\n{bcolors.OKBLUE}<{agent_name} Prompt>{bcolors.ENDC}")
            print(user_input)
        self._messages.append({"role": "user", "content": user_input})
        try:
            completion = self._chat_completion()
            assistant_reply = completion.choices[0].message.content
            self._messages.append({"role": "assistant", "content": assistant_reply})
            if verbose >= 1:
                print(f"\n{bcolors.OKGREEN}<{agent_name} Response>{bcolors.ENDC}")
                print(f"{bcolors.OKGREEN}{assistant_reply}{bcolors.ENDC}")
            return assistant_reply
        except OpenAIError as e:
            error_str = str(e)
            if "context_length_exceeded" in error_str:
                print("Context length exceeded. Here is the conversation context:")
                pprint(self._messages)
            print(f"Rate limit exceeded: {e}")
            raise
