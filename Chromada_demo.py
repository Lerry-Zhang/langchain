# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 22:05
# @Author  : lerry zhang
# @File    : Chromada_demo.py
import numpy as np
import pandas as pd
import json
from config.ConfigurationManager import load
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain import hub
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class TestStep:
    def __init__(self, test_step, test_step_description, test_step_expected):
        self.test_step = test_step
        self.test_step_description = test_step_description
        self.test_step_expected = test_step_expected

    def jsonFormat(self):
        return {
            "test_step": self.test_step,
            "test_step_description": self.test_step_description,
            "test_step_expected": self.test_step_expected
        }

class TestPlan:
    def __init__(self, test_name, application_name, country):
        self.test_name = test_name
        self.application_name = application_name
        self.country = country
        self.test_steps = []

    def add_test_step(self, test_step: TestStep) -> TestStep:
        self.test_steps.append(test_step.jsonFormat())

    def toJsonFormat(self):
        dict = {
            "Test_Name": self.test_name,
            "Application": self.application_name,
            "Country": self.country,
            "Test_Steps": self.test_steps
        }

        return json.dumps(dict)


if __name__ == '__main__':
    load()

    excel_file_path = r"testCases.xlsx"
    data = pd.read_excel(excel_file_path)

    list = []
    for i in data.index.values:
        line = data.loc[i, ['Test Name', 'Application Name','Test Country','Test Step','Test Step Description','Test Step Expected']].to_dict()
        list.append(line)

    test_plan_list = []
    test_plan = TestPlan("", "", "")
    for row in list:
        test_step = TestStep(test_step=row["Test Step"], test_step_description=row["Test Step Description"],
                             test_step_expected=row["Test Step Expected"])

        if  pd.isna(row["Test Name"]):
            test_plan.add_test_step(test_step)
        else:
            if test_plan.test_name != "":
                test_plan_list.append(test_plan)

            test_plan = TestPlan(test_name=row["Test Name"], application_name=row["Application Name"],
                                 country=row["Test Country"])

            test_plan.add_test_step(test_step)

    test_plan_list.append(test_plan)

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    document_list = []
    texts = []
    for case in test_plan_list:
        print(case.toJsonFormat())
        doc = Document(page_content=case.toJsonFormat())
        # splits = text_splitter.split_text(case.toJsonFormat())
        document_list.append(doc)
        # texts += (splits)

    # vectorstore = Chroma.from_texts(texts=texts, embedding=OpenAIEmbeddings(), persist_directory=r"chroma")
    vectorstore = Chroma.from_documents(documents=document_list, embedding=OpenAIEmbeddings(), persist_directory=r"chroma")
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print(retriever.invoke("’client', 'transfer', 'customer', 'account'" ))
    print("\r\n")
    print(prompt)
    print("\r\n")
    resp = rag_chain.invoke("Client transfer to another customer account")

    print(resp)
