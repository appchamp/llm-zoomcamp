```python
# !wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py
```


```python
import minsearch
```


```python
import json
import tqdm
```


```python
with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)
```


```python
documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)
```


```python
documents[0]
```




    {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
     'section': 'General course-related questions',
     'question': 'Course - When will the course start?',
     'course': 'data-engineering-zoomcamp'}




```python
index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
```

SELECT * WHERE course = 'data-engineering-zoomcamp';


```python
q = 'the course has already started, can I still enroll?'
```


```python
index.fit(documents)
```




    <minsearch.Index at 0x70035ed7eef0>




```python
from openai import OpenAI
```


```python
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)
```


```python
response = client.chat.completions.create(
    model='phi3',
    messages=[{"role": "user", "content": q}]
)

response.choices[0].message.content
```




    " Yes, typically you can still enroll in a course that has already started. However, to provide an accurate answer and assistance, it's important to check with the specific institution or instructor offering the course for their policies on late enrollment. Some courses may have limited space and cannot accommodate additional students once they begin, while others might allow you to join if there is available room in class capacity. It's also advisable to consider any potential impacts on your learning experience due to joining later than planned."




```python
def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )

    return results
```


```python
def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt
```


```python
def llm(prompt):
    response = client.chat.completions.create(
        model='phi3',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```


```python
query = 'how do I run kafka?'

def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
```


```python
search_results = search(query)
prompt = build_prompt(query, search_results)
print(prompt)
```

    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.
    
    QUESTION: How do I execute a command in a running docker container?
    
    CONTEXT: 
    section: Module 1: Docker and Terraform
    question: PGCLI - running in a Docker container
    answer: In case running pgcli  locally causes issues or you do not want to install it locally you can use it running in a Docker container instead.
    Below the usage with values used in the videos of the course for:
    network name (docker network)
    postgres related variables for pgcli
    Hostname
    Username
    Port
    Database name
    $ docker run -it --rm --network pg-network ai2ys/dockerized-pgcli:4.0.1
    175dd47cda07:/# pgcli -h pg-database -U root -p 5432 -d ny_taxi
    Password for root:
    Server: PostgreSQL 16.1 (Debian 16.1-1.pgdg120+1)
    Version: 4.0.1
    Home: http://pgcli.com
    root@pg-database:ny_taxi> \dt
    +--------+------------------+-------+-------+
    | Schema | Name             | Type  | Owner |
    |--------+------------------+-------+-------|
    | public | yellow_taxi_data | table | root  |
    +--------+------------------+-------+-------+
    SELECT 1
    Time: 0.009s
    root@pg-database:ny_taxi>
    
    section: Module 1: Docker and Terraform
    question: Docker - Error response from daemon: Conflict. The container name "pg-database" is already in use by container “xxx”.  You have to remove (or rename) that container to be able to reuse that name.
    answer: Sometimes, when you try to restart a docker image configured with a network name, the above message appears. In this case, use the following command with the appropriate container name:
    >>> If the container is running state, use docker stop <container_name>
    >>> then, docker rm pg-database
    Or use docker start instead of docker run in order to restart the docker image without removing it.
    
    section: Module 1: Docker and Terraform
    question: Docker - Cannot pip install on Docker container (Windows)
    answer: You may have this error:
    Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.u
    rllib3.connection.HTTPSConnection object at 0x7efe331cf790>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')':
    /simple/pandas/
    Possible solution might be:
    $ winpty docker run -it --dns=8.8.8.8 --entrypoint=bash python:3.9
    
    section: Module 1: Docker and Terraform
    question: Docker - Cannot connect to Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
    answer: Make sure you're able to start the Docker daemon, and check the issue immediately down below:
    And don’t forget to update the wsl in powershell the  command is wsl –update
    
    section: Module 6: streaming with kafka
    question: How do I check compatibility of local and container Spark versions?
    answer: You can check the version of your local spark using spark-submit --version. In the build.sh file of the Python folder, make sure that SPARK_VERSION matches your local version. Similarly, make sure the pyspark you pip installed also matches this version.



```python
rag(query)
```




    " To run Kafka in the terminal, you would typically create a virtual environment and then execute your Java or Python scripts within that environment. For example:\n\n\n1. **Java Kafka**: Firstly, ensure your project directory is set up correctly with necessary dependencies (build/libs JAR file). Then, navigate to your project's terminal and run the following command to start a Java producer:\n\n   ```java\n\n   java -cp build/libs/<your_jar_name>-1.0-SNAPSHOT.jar:out src/main/java/org/example/JsonProducer.java\n\n   ```\n\n\nPlease replace `<your_jar_name>` with the actual JAR filename used for your project and adjust `src/main/java/org/example/JsonProducer.java` to point at your Java Kafka producer class file as necessary.\n Writen by an AI model, this information is based on the given context and should not be considered exhaustive or universally applicable."




```python
rag('the course has already started, can I still enroll?')
```




    " Yes, even if you don't register before the start date of the course (which is January 15th, 2024 at 17:00), you are still eligible to submit homework assignments and follow the course materials after its completion for self-study. However, please be mindful that there will be deadlines for final project submissions during the course."




```python
documents[0]
```




    {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
     'section': 'General course-related questions',
     'question': 'Course - When will the course start?',
     'course': 'data-engineering-zoomcamp'}




```python
from elasticsearch import Elasticsearch
```


```python
es_client = Elasticsearch('http://localhost:9200') 
```


```python
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}

index_name = "course-questions"

es_client.indices.create(index=index_name, body=index_settings)
```




    ObjectApiResponse({'acknowledged': True, 'shards_acknowledged': True, 'index': 'course-questions'})




```python
documents[0]
```




    {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
     'section': 'General course-related questions',
     'question': 'Course - When will the course start?',
     'course': 'data-engineering-zoomcamp'}




```python
from tqdm.auto import tqdm
```


```python
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
```


      0%|          | 0/948 [00:00<?, ?it/s]



```python
query = 'I just disovered the course. Can I still join it?'
```


```python
def elastic_search(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs
```


```python
def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
```


```python
rag(query)
```




    " Yes, you can still join the course even if you discover it after the start date. However, there will be deadlines for turning in final projects so it's best not to leave everything for the last minute. Additionally, we keep all materials after the course finishes, allowing you to follow the course at your own pace and continue preparing for future cohorts."




```python
import requests 

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```


```python
query = 'How do I execute a command in a running docker container?'

search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }


```


```python
response = es_client.search(index=index_name, body=search_query)

```


```python
response['hits']['hits'][0]['_score']

```




    75.54128




```python
query = 'How do I execute a command in a running docker container?'

search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }


```


```python
response = es_client.search(index=index_name, body=search_query)
result_docs = []
    
for hit in response['hits']['hits']:
    result_docs.append(hit['_source'])
```


```python
result_docs[2]['question']
```




    'How do I copy files from a different folder into docker container’s working directory?'




```python
def build_prompt_Q5(query, search_results):
    prompt_template_Q5 = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

    context_template_Q5 = """
Q: {question}
A: {text}
""".strip() + "\n\n"

    context = ""
    for doc in search_results:
        context = context + context_template_Q5.format(question=doc['question'], text=doc['text'])
        
    prompt = prompt_template_Q5.format(question=query, context=context).strip()
    return prompt
```


```python
prompt = build_prompt_Q5(query, result_docs)
print(len(prompt))

```

    2710



```python
!pip install tiktoken
```

    Collecting tiktoken
      Downloading tiktoken-0.7.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
    Collecting regex>=2022.1.18 (from tiktoken)
      Downloading regex-2024.5.15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (40 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m40.9/40.9 kB[0m [31m343.6 kB/s[0m eta [36m0:00:00[0mMB/s[0m eta [36m0:00:01[0m
    [?25hRequirement already satisfied: requests>=2.26.0 in /home/codespace/.local/lib/python3.10/site-packages (from tiktoken) (2.32.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /home/codespace/.local/lib/python3.10/site-packages (from requests>=2.26.0->tiktoken) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /home/codespace/.local/lib/python3.10/site-packages (from requests>=2.26.0->tiktoken) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/python/3.10.13/lib/python3.10/site-packages (from requests>=2.26.0->tiktoken) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /home/codespace/.local/lib/python3.10/site-packages (from requests>=2.26.0->tiktoken) (2024.2.2)
    Downloading tiktoken-0.7.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m3.6 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m0:01[0m:01[0m
    [?25hDownloading regex-2024.5.15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (775 kB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m775.1/775.1 kB[0m [31m5.0 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m0:01[0mm
    [?25hInstalling collected packages: regex, tiktoken
    Successfully installed regex-2024.5.15 tiktoken-0.7.0



```python
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")
print(len(encoding.encode(prompt)))
```

    621

