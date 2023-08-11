### Siri-GPT

 中文 / [English](./README_EN.md)
#### 介绍
这是通过apple快捷指令与chatgpt交互的解决方案

快捷指令下载地址：https://www.icloud.com/shortcuts/a7173900a84142f3b7a0a556c21fc465

项目需要用到 OpenAI API key

项目基于[langchain](https://github.com/langchain-ai/langchain)，支持记录上下文，每次对话开始时算一轮对话，如果一轮对话的超出token限制会根据设置的token长度进行截取

#### 使用方法
1. 部署项目（项目需要用到redis，需先部署redis，或者使用docker compose部署）
2. 下载安装[苹果快捷指令](https://www.icloud.com/shortcuts/a7173900a84142f3b7a0a556c21fc465)并配置相关参数：  
     - host: 服务端地址，http://ip_address:port  
     - token: 访问令牌，部署项目时填写的API_KEY
3. 在苹果设备上手动运行一次“对话”快捷指令，同意所有权限。
4. “Hey Siri，对话”，开始与ChatGPT对话。

#### 依赖

- Redis

- Python

#### 本地运行

1. 配置环境变量或者直接写在项目根目录的.env文件，参考.env.example

2. pip install -r requirements.txt

3. gunicorn -c gunicorn_config.py app:app

##### Docker部署
```
# Run in the background
docker run -d -p 5000:5000 \
   -e OPENAI_API_KEY="sk-xxxxxxx" \
   -e API_KEY="ak-xxxxxx,ak-xxxxxxx" \
   -e PROXY_URL="http://xxxxxx:xxxx" \
   -e REDIS_HOST="xxxxxxx"  \
   -e REDIS_PORT=6379  \
   iweus/siri-gpt

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



