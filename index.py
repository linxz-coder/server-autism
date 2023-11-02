# app.py
from flask import Flask, request, Response
import os
import re
from dotenv import load_dotenv
from flask_cors import CORS
import openai
import pinecone
from openai.embeddings_utils import get_embedding

load_dotenv()  # 加载 .env 文件中的变量

app = Flask(__name__)
CORS(app, origins="*")
openai.api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量中获取 API 密钥

@app.route("/api/python", methods=["POST"])
def generate():
    content = request.json.get('content')
    chatHistory = request.json.get('chatHistory')
    print("chatHistory: " + chatHistory)

    content_with_chatHistory = f"""你的名字是豆豆，正在与对方沟通。
    之前的对话:
    {chatHistory}

    对方新提出的问题: {content}
    你的回复:"""

    def autism_expert(question):
        """当用户问自闭症问题时，搜索专业答案"""
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
        index = pinecone.Index('openai')
        query_embed = get_embedding(question, engine='text-embedding-ada-002')
        result = index.query(vector=query_embed, top_k=3, include_metadata=True)
        result_text = result['matches'][0].metadata['text']

        return result_text

    def run_conversation():
        # Step 1: send the conversation and available functions to GPT
        messages = [{"role": "system", "content": "你是自闭症的康复专家"},
                    {"role": "user", "content": content_with_chatHistory}]
        functions = [
            {
                "name": "autism_expert",
                "description": "当用户咨询自闭症类问题有用",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "用户的问题",
                        },
                    },
                    "required": ["question"],
                },
            }
        ]

        flag_executed = False

        for res in openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                functions=functions,
                function_call="auto",  # auto is default, but we'll be explicit
                stream=True
        ):
            # print("res: ")
            # print(res)
            # print("====================")
            delta = res.choices[0].delta
            # print("delta: ")
            # print(delta)
            # print("====================")

            if "function_call" not in delta and 'content' in delta:
                # print("delta.content")
                # print(delta.content)
                # print("====================")
                if 'content' in delta:
                    yield delta.content
                    flag_executed = True

        if not flag_executed:
            function_name = "autism_expert"
            fuction_to_call = autism_expert
            function_response = fuction_to_call(
                question=content,
            )

            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
            print("function_message: ")
            print(messages)
            print("====================")

            search_result = messages[2]['content']
            # print("search_result: " + search_result)
            answer_match = re.search(r'answer:(.*)', search_result, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1)
                print("answer: " + answer)
                print("====================")

            prompt = f"""
            根据用户的问题，参考背景信息，输出回答。要求回答简短切题，且尽可能有趣地回复。用户问题和参考背景信息会用{{}}来表示。
            用户问题：{content}, 参考背景信息：{answer}
            """

            for chunk in openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=0,
                    stream=True  # this time, we set stream=True
            ):
                # 有的chunk没有content
                if 'content' in chunk.choices[0].delta:
                    # print(chunk.choices[0].delta.content)
                    yield chunk.choices[0].delta.content

    return Response(run_conversation(), mimetype='text/event-stream')


@app.route("/api/title", methods=["POST"])
def generate_title():
    content = request.json.get('content')

    title_prompt = f'''
    使用不多于四个字来直接返回这段对话的主题作为标题，不要解释、不要标点、不要语气词、不要多余文本，如果没有主题，请直接返回“闲聊”
    {content}
    '''

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=title_prompt,
        temperature=0,
    )

    return response.choices[0].text


if __name__ == '__main__':
    # app.run(port=5328,debug=True)
    app.run(host='0.0.0.0', port=5328, debug=True)
