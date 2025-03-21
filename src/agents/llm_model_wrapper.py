import os
import backoff
import openai
from dotenv import load_dotenv
from openai import OpenAIError

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class LLMModelWrapper:
    def __init__(
        self,
        system_message,
        model="gpt-4o",
        temperature=0.6,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        self._model = model
        self._messages = [{"role": "system", "content": system_message}]
        self._temperature=temperature
        self._top_p=top_p
        self._frequency_penalty=frequency_penalty
        self._presence_penalty=presence_penalty

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=60)
    def _chat_completion(self):
        return openai.chat.completions.create(
            model=self._model,
            messages=self._messages,
            temperature=0.6,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

    def step(self, user_input):
        self._messages.append({"role": "user", "content": user_input})
        try:
            completion = self._chat_completion()
            assistant_reply = completion.choices[0].message.content
            self._messages.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply
        except OpenAIError as e:
            print(f"Rate limit exceeded: {e}")
            raise
