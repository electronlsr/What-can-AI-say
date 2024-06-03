from pymilvus import MilvusClient
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from xpinyin import Pinyin
import dashscope
import random
from typing import Generator
import requests
from urllib import parse
import uuid
import os

collection_name = "baidu_competition"
EMBEDDING_DEVICE = "cpu"
dashscope.api_key = 'sk-0793fab51dbd44f1a3dbf2e0541990f9'
p = Pinyin()
limit = 2  
history = []
all_query = ""
embeddings = HuggingFaceEmbeddings(model_name="./m3e-base", model_kwargs={'device': EMBEDDING_DEVICE})
prompt = '### 角色与目标\n* 你是一个**专业的中式厨师和营养师**，精通国内各种菜系和各地菜肴，你的知识库里也有庞大的菜谱。\n* 你可以根据给定的食材生成菜谱。一定要用到提供的食材，尽量用完食材，可以额外补充一些常见的家用食材，尽量满足用户的个人口味需求。一定要针对菜肴给出烹饪难度评分和需要的食材、工具。\n* 你可以根据个人需求定制一顿餐的菜谱。如果对方有减脂管理、备孕怀胎、营养问题等特殊情况或需求，你需要根据需求量身定制一顿营养全面的菜谱（具体到特定的菜肴）并给出理由，随后可以对每道菜的具体做法进行介绍。\n* 用户直接询问一道菜的做法时，严格按照格式要求回答。\n5. 涉及复杂的烹饪技巧或方法或用户询问，要给出具体做法的提示。\n### 指导原则\n* 专业性：确保提供的菜谱和营养建议基于专业的烹饪和营养学知识或知识库内容。\n* 友善性：在回答用户时，保持友善和尊重。\n### 限制\n* 尽量避免使用“适量”，“少许”等不明确的词汇。\n* 如果遇到不明白或者不合适的菜肴，请不要回答相关问题。\n* 如果用户有忌口，请一定不要给出相关菜肴。\n* 提供菜谱或具体做法时，你必须**严格按照格式要求**进行回答。\n* 对于不适合一起烹饪的食材一定要多设计几道菜肴。不要给出过于少见的菜肴。\n### 澄清\n* 你只能回答关于中式烹饪、菜谱生成、烹饪技巧、营养定制、菜肴做法难易度评分以及菜单生成等方面的问题。\n* 你有专业的中国八大菜系相关知识，提及有关内容可以从知识库检索作答。\n* 对于要准备的食材，尽可能明确给出需要的量，必要时可以进行分类讨论。\n### 格式要求与示例\n\n--- 我有茄子豆角和五花肉，可以做什么？ / 豆角茄子炒五花肉怎么做？\n--- 您可以制作一道美味的豆角茄子炒五花肉。以下是菜谱：\n**豆角茄子炒五花肉**\n**烹饪难度**：★★★☆☆ <给出你的难度评分>\n**所需食材**：<提供所需食材>\n**所需工具**：<提供所需工具>\n**步骤**：<提供具体步骤>\n**制作时长**： 准备时间：约15分钟。烹饪时间：约半小时 <给出你的时长估计>\n**提示**：\n这道菜色香味俱全，五花肉的鲜香与豆角和茄子的清甜完美融合，非常适合作为家常菜来享用。希望您能喜欢！\n\n--- 我正在减肥，午饭怎么吃？\n--- 若您正在减肥，午饭的饮食应该以低热量、高蛋白、高纤维的食物为主。下面是一份简单又营养的减肥午餐食谱，供您参考：\n**减肥午餐食谱**\n**主食**：\n**蛋白质**：\n**蔬菜**：\n**汤品**：\n这份午餐食谱的热量控制在400-500大卡左右。'

client = MilvusClient(
    uri="http://114.212.97.40:19530",
)

def recipe_api(name : str) -> str:
    search = name
    search = "search?keyword=" + parse.quote(search) + "&num=10&start=1"
    host = "http://jsucpdq.market.alicloudapi.com/recipe/"
    url = parse.urljoin(host, search)
    header = {
        "Authorization": "APPCODE 79942dcf76664fb59347207c63b9593e"
    }

    try:
        ans = requests.get(url=url, headers=header)
        data = json.loads(ans.text)
        data = data["result"]["list"][0]
        string = ""
        string += "菜名：" + data['name'] + "\n" + "预计用时:" + data['cookingtime'] + '\n' + '原料:' + '\n'
        for item in data['material']:
            string += item['amount'] + " " + item['mname'] + '\n'
        string += "步骤:" + '\n'
        for item in data["process"]:
            string += item["pcontent"] + "\n"
        if string == "":
            return "None"
        else:
            return string
    except:
        return "None"

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

def partition_search_api(messages):
    query = messages[-1]['content']
    message = [
    {'role': 'user', 'content': f'你是一名问题分类员，请你根据下面的问题告诉我用户想要询问的是烹饪技巧还是菜谱还是定制食谱还是打招呼，只需要告诉我类别即可，不要返回其他任何东西。例如，若问题是：土豆烧肉怎么做？请返回：菜谱。若问题是：怎么腌制猪肉？请返回：烹饪技巧。若问题是：我不吃辣，想吃点海鲜，可以吃什么？请返回：定制食谱。若问题是：你好。请返回：打招呼。除此之外都请返回"未知"。接下来请你对下面的问题进行分类：{query}'},]
    response = dashscope.Generation.call(
        'qwen1.5-32b-chat',
        messages=message,
        seed=random.randint(1, 10000),
        result_format='text',
    )
    res = json.loads(str(response))['output']['text']
    partition = []
    print(res)
    if res.find('烹饪技巧') != -1:
        embedding = embeddings.embed_documents([query])
        results = client.search(collection_name=collection_name, data=embedding, limit=limit, search_params={"metric_type": "COSINE", "params": {}}, output_fields=["file_path"], partition_names=['pengrenjiqiao'])
        return [result['entity']['file_path'] for result in results[0]]
    elif res.find('菜谱') != -1:
        queries = ""
        for message in messages:
            if message['role'] == 'user':
                queries += message['content']
        message = [
            {'role': 'user', 'content': f'你是一个**专业的中式厨师和营养师**，精通国内各种菜系和各地菜肴，请你根据下面的提问返回你能想到的所有的相关菜肴的名称，在中括号中给出，用逗号隔开。例如，当我问"土豆猪肉能做什么"请你返回["土豆烧肉", "土豆丝炒肉"]。请你针对下面的问题进行回答：{queries}'},]
        response = dashscope.Generation.call(
            'qwen1.5-32b-chat',
            messages=message,
            seed=random.randint(1, 10000),
            result_format='text',
        )
        res = json.loads(str(response))['output']['text']
        if not is_json(res):
            embedding = embeddings.embed_documents([query])
            results = client.search(collection_name=collection_name, data=embedding, limit=limit, search_params={"metric_type": "COSINE", "params": {}}, output_fields=["file_path"], partition_names=['caipu'])
            return [result['entity']['file_path'] for result in results[0]]
        res = json.loads(res)
        resf = []
        for item in res:
            embedding = embeddings.embed_documents([item])
            results = client.search(collection_name=collection_name, data=embedding, limit=1, search_params={"metric_type": "COSINE", "params": {}}, output_fields=["file_path"], partition_names=['caipu'])
            resf += [result['entity']['file_path'] for result in results[0]]
            apires = recipe_api(item)
            if not os.path.exists('temp'):
                os.mkdir('temp')
            if apires != "None":
                tempf = 'temp/' + str(uuid.uuid4()) + '.txt'
                with open(tempf, 'w', encoding='utf-8') as f:
                    f.write(apires)
                resf.append(tempf)
        return resf
    elif res.find('定制食谱') != -1:
        queries = ""
        for message in messages:
            if message['role'] == 'user':
                queries += message['content']
        message = [
            {'role': 'user', 'content': f'你是一个**专业的中式厨师和营养师**，精通国内各种菜系和各地菜肴，请你根据下面的提问返回你能想到的所有的相关菜肴的名称，在中括号中给出，用逗号隔开。例如，当我问"我最近想减肥，能吃什么"你可以返回["蔬菜沙拉", "水煮鸡胸肉"]或者类似的答案。请你针对下面的问题进行回答：{queries}'},]
        response = dashscope.Generation.call(
            'qwen1.5-32b-chat',
            messages=message,
            seed=random.randint(1, 10000),
            result_format='text',
        )
        res = json.loads(str(response))['output']['text']
        if not is_json(res):
            embedding = embeddings.embed_documents([query])
            results = client.search(collection_name=collection_name, data=embedding, limit=limit, search_params={"metric_type": "COSINE", "params": {}}, output_fields=["file_path"], partition_names=['caipu'])
            return [result['entity']['file_path'] for result in results[0]]
        res = json.loads(res)
        resf = []
        for item in res:
            embedding = embeddings.embed_documents([item])
            results = client.search(collection_name=collection_name, data=embedding, limit=1, search_params={"metric_type": "COSINE", "params": {}}, output_fields=["file_path"], partition_names=['caipu'])
            resf += [result['entity']['file_path'] for result in results[0]]
            apires = recipe_api(item)
            if not os.path.exists('temp'):
                os.mkdir('temp')
            if apires != "None":
                tempf = 'temp/' + str(uuid.uuid4()) + '.txt'
                with open(tempf, 'w', encoding='utf-8') as f:
                    f.write(apires)
                resf.append(tempf)
        return resf
    elif res.find('打招呼') != -1:
        return []
    else:
        embedding = embeddings.embed_documents([query])
        results = client.search(collection_name=collection_name, data=embedding, limit=limit, search_params={"metric_type": "COSINE", "params": {}}, output_fields=["file_path"])
        return [result['entity']['file_path'] for result in results[0]]
        
    
def ask_stream(messages) -> Generator:
    if messages[-1]["content"] == "已有食材怎么制作":
        yield "请告诉我您目前有哪些可以用的食材，我会根据这些食材为你生成合适的菜谱。如果有一些口味需求也可以告诉我！"
        return
    if messages[-1]["content"] == "我想定制一个食谱":
        yield "请告诉我您有什么需求或忌口，我会尽力给出一个满足需求的食谱。"
        return
    if messages[-1]["content"] == "能教我做一道菜吗":
        yield "请问您想要我提供哪一道菜的菜谱？"
        return
    if messages[-1]["content"] == "我想学习烹饪技巧":
        yield "请问您具体想要了解哪种烹饪技巧？或者在哪一步遇到了困难？"
        return
    files = partition_search_api(messages)
    if not files:
        yield "您好，我是中式饮食专家。很高兴为您提供专业的中式烹饪方法和营养建议。无论您是想了解特定菜肴的做法，还是需要个性化的饮食方案，又或者想了解某个菜系、某种烹饪方法，我都能为您提供详尽的指导。让我们一起探索美食吧！\n试试问我：已有食材怎么制作 我想定制一个食谱 能教我做一道菜吗 我想学习烹饪技巧"
        return
    files_content = ""
    for file in files:
        with open(file.replace('\\','/'), 'r', encoding='utf-8') as f:
            files_content += f.read()
    query_with_file = [
        {'role': 'user', 'content': f'{files_content}\n请你将以上提供的额外内容作为补充知识回答这个问题：{messages[-1]["content"]}\n如果提供的额外内容和问题关系不大，请直接忽视提供的额外内容直接回答用户的问题。不管我提供了什么额外内容，你在回答时都禁止直接提及提供的额外内容，即禁止说类似“根据你提供的信息”“很抱歉，你提供的信息和问题无关”这样的语句，只专注于回答用户的问题，不要被额外信息干扰。'},]
    responses = dashscope.Generation.call(
        'qwen1.5-32b-chat',
        messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': '你好'}] + messages[:-1] + query_with_file,
        seed=random.randint(1, 10000),
        result_format='text',
        stream=True,
        incremental_output=True,
    )
    for response in responses:
        #if not is_json(str(response)):
        # print(str(response))
        # return
        # yield str([{'role': 'system', 'content': prompt}] + messages[:-1] + query_with_file) + str(response)
        yield json.loads(str(response))['output']['text']  
