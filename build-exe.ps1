python -m pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m PyInstaller -n siri-gpt --onefile --collect-data langchain -c ./app.py