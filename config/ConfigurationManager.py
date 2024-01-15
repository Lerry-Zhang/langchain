# -*- coding: utf-8 -*-
# @Time    : 2023/12/19 12:14
# @Author  : lerry zhang
# @File    : ConfigurationManager.py
def load():
    import dotenv
    dotenv.load_dotenv(override=True)


if __name__ == '__main__':
    load()
    import os
    environ = os.environ
    openai_api_key = environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        print('Open AI API Key not found')
    else:
        print('OPENAI_API_KEY=' + environ.get('OPENAI_API_KEY'))


