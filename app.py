from flask import Flask, request, jsonify
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from flask import Flask, request
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Union, Dict, List
import secrets
import threading
import re
import os
import redis
import logging
from logging.handlers import TimedRotatingFileHandler


app = Flask(__name__)


MIN_LENGTH = 20

END_MARK = "<END>"

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler('app.log', when='midnight', interval=1, backupCount=7)
handler.suffix = '%Y-%m-%d'
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


API_KEY = os.getenv("API_KEY")
openai.proxy = os.getenv("PROXY_URL")
openai.api_base = os.getenv("OPEN_AI_BASE")

MODEL = os.environ.get("MODEL","gpt-3.5-turbo")
MAX_TOKEN_LIMIT = int(os.environ.get('MAX_TOKEN_LIMIT', 2000))

REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_PASSWORD = os.environ.get('REDIS_AUTH', None)
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
SYSTEM_TEMPLATE = os.environ.get('SYSTEM_TEMPLATE', "You are a nice chatbot having a conversation with a person.") 

redis_client = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT,
    password=REDIS_PASSWORD, 
    db=REDIS_DB)

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

def is_sentence_ended(text):
    punctuation_marks = [".", "?", "!", "...", ";", ":", "。", "？", "！", "……", "；", "："]
    last_char = text[-1]
    return last_char in punctuation_marks


def is_token_auth(token):
    if not token or not API_KEY:
        return False
    keys = API_KEY.split(",")
    return token in keys


def split_text(text):
    if not text:
        return "", None
    
    text = text.lstrip()
    sentences = cut_sent(text)
    
    if len(sentences) == 1:
        if is_sentence_ended(text):
            return text, ""
        return "", None
    
    last_sentence = sentences[-1]
    if not is_sentence_ended(last_sentence):
        sentences.pop()
    
    total_length = 0
    for i, sentence in enumerate(sentences):
        total_length += len(sentence)
        if total_length >= MIN_LENGTH:
            break
    
    if total_length >= MIN_LENGTH:
        message = text[:total_length]
        remaining_text = text[total_length:]
        return message, remaining_text
    
    return "", None

def append_text_to_redis(redis_key, text):
    content = redis_client.get(redis_key)
    if content is not None:
        content = content.decode("utf-8") + text
    else:
        content = text
    redis_client.set(redis_key, content)

def get_redis_key(session_id,question_id):
    return f"{session_id}_{question_id}"

class CustomTokenMemory(BaseChatMemory):
    new_buffer: List = []
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm:BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        if not self. new_buffer:
            self. prune_memory()
        return self. new_buffer

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self. memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer: Any = self.buffer
        if self. return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self. memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer. Pruned."""
        super(). save_context(inputs, outputs)
        self. prune_memory()
        
    def prune_memory(self):
        # Prune buffer if it exceeds max token limit
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            while curr_buffer_length > self.max_token_limit:
                buffer. pop(0)
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        self.new_buffer = buffer

class StreamingGradioCallbackHandler(BaseCallbackHandler):
    
    def __init__(self,redis_key):
        self.redis_key = redis_key
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        append_text_to_redis(self.redis_key,token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        append_text_to_redis(self.redis_key,END_MARK)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        append_text_to_redis(self.redis_key,END_MARK)
    
def ask_question(session_id,question_id,question):
    redis_key = get_redis_key(session_id,question_id)
    callbackhandler = StreamingGradioCallbackHandler(redis_key)
    llm = ChatOpenAI(streaming=True,callbacks=[callbackhandler])
    redis_url = f'redis://{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    message_history = RedisChatMessageHistory(url=redis_url, ttl=600, session_id=session_id)
    memory = CustomTokenMemory(llm=llm,max_token_limit=MAX_TOKEN_LIMIT,memory_key="chat_history", return_messages=True,chat_memory=message_history)
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                SYSTEM_TEMPLATE
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    resp = chain({"question": question})
    answer = resp["text"]
    logger.info(f"{session_id} answer: {answer}")
    
@app.route('/api/ask', methods=['POST'])
def ask():
    token = request.headers.get('token')
    if not is_token_auth(token=token):
        return jsonify({'status': 'fail'})
    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id')
    question_id = secrets.token_hex(8)
    if session_id is None or len(session_id) == 0:
        session_id = secrets.token_hex(16)
    logger.info(f"{session_id} question: {question}")
    redis_key = get_redis_key(session_id,question_id)
    redis_client.set(redis_key,"")
    thread = threading.Thread(target=ask_question, args=(session_id,question_id,question,))
    thread.start()
    return jsonify({'status': 'ok','session_id': session_id,'question_id': question_id})

@app.route('/api/answer', methods=['POST'])
def answer():
    token = request.headers.get('token')
    if not is_token_auth(token=token):
        return jsonify({'status': 'fail'})
    data = request.get_json()
    session_id = data.get('session_id')
    question_id = data.get('question_id')
    if session_id is None or question_id is None:
        return jsonify({'status': 'fail'})
    redis_key = get_redis_key(session_id,question_id)
    msg = redis_client.get(redis_key)
    if msg is not None:
        msg = msg.decode("utf-8")
        if msg.endswith(END_MARK):
            redis_client.delete(redis_key)
            msg = msg.rstrip(END_MARK)
            resp = jsonify({'status': 'end', 'msg': msg})
        else:
            value1,value2 = split_text(msg)
            resp = jsonify({'status': 'running', 'msg': value1})
            if value2 is not None:
                redis_client.set(redis_key,value2)
    else:
        resp = jsonify({'status': 'end'})
    return resp

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=5000)
