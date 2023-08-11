### Siri-GPT

English / [简体中文](./README_CN.md)


##### Introduction

Siri-GPT is a solution for interacting with ChatGPT using Apple Shortcuts.

Shortcut Download Link: https://www.icloud.com/shortcuts/a7173900a84142f3b7a0a556c21fc465

The project requires an OpenAI API key.

The project is based on [langchain](https://github.com/langchain-ai/langchain) and supports context retention. Each conversation is treated as a separate round, and if a round exceeds the token limit, it is truncated based on the specified token length.


##### Usage

1. Deploy the project (requires Redis, so deploy Redis first).
2. Download and install the [Apple Shortcuts](https://www.icloud.com/shortcuts/a7173900a84142f3b7a0a556c21fc465) and configure the following parameters:  
    - host: Server address (e.g., http://ip_address:port)  
    - token: Access token (API_KEY used during project deployment)
3. Manually run the "对话" shortcut on your Apple device and grant all permissions.


##### Dependencies

- Redis
- Python

##### Running the Project
1. Set up the environment variables or add them directly to the .env file in the project's root directory (refer to .env.example for reference).
2. Install the required dependencies: pip install -r requirements.txt.
3. Run the project using Gunicorn: gunicorn -c gunicorn_config.py app:app.

##### Environment Variables

- OPENAI_API_KEY (required): OpenAI API key obtained from your OpenAI account page.

- API_KEY (required): Access token(s), optional but can include multiple tokens separated by commas. Corresponds to the token in the shortcut.

- MODEL (optional): OpenAI language model, default is gpt-3.5-turbo.

- MAX_TOKEN_LIMIT (optional): OpenAI token limit, default is 2000.

- PROXY_URL (optional): OpenAI HTTP proxy address.

- REDIS_HOST (required): Redis address.

- REDIS_PORT (required): Redis port.


##### Docker build & Run
```
docker build -t siri-gpt .

# Run in the background
docker run -d -p 5000:5000 \
   -e OPENAI_API_KEY="sk-xxxxxxx" \
   -e API_KEY="ak-xxxxxx,ak-xxxxxxx" \
   -e PROXY_URL="http://xxxxxx:xxxx" \
   -e REDIS_HOST="xxxxxxx"  \
   -e REDIS_PORT=6379  \
   iweus/siri-gpt

# Access the application at
http://localhost:5000/
```



##### Docker compose

```yaml
version: "3"

services:
  siri-gpt:
    image: iweus/siri-gpt 
    environment:
      - API_KEY=xxxxxx
      - OPENAI_API_KEY=xxxxxx
      - PROXY_URL=xxxx
      - REDIS_HOST=xxxx
      - MAX_TOKEN_LIMIT=4000
      - REDIS_PORT=6379
      - MODEL=gpt-3.5-turbo
    ports:
      - 5000:5000
    depends_on:
      - redis

  redis:
    image: redis
    ports:
      - 6379:6379
```