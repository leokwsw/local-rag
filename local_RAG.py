import subprocess
import os
import sys
from typing import List, Dict
import whisper
from openai import OpenAI
import uuid
import weaviate
from weaviate.util import get_valid_uuid
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import DataSourceMetadata
from unstructured.partition.json import partition_json
from sentence_transformers import SentenceTransformer
from langchain.llms import LlamaCpp
from langchain.vectorstores.weaviate import Weaviate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import tiktoken
from colors import color
from dotenv import load_dotenv
import mimetypes

mimetypes.init()

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")

if openai_key == "<YOUR_OPENAI_KEY>":
    openai_key = ""

if openai_key == "":
    sys.exit("Please Provide Your OpenAI API Key")

output_dir = 'my-docs'
input_dir = 'files_used'
weaviate_url = "http://localhost:8080"
embedding_model_name = 'all-MiniLM-L6-v2'
device = 'mps'

root = os.path.dirname(__file__)
use_verbose = True

whisper_model = whisper.load_model("tiny", device="cpu", download_root="model_files")

use_openai = False


def process_local(output_dir: str, num_processes: int, input_path: str):
    command = [
        "unstructured-ingest",
        "local",
        "--input-path", input_path,
        "--output-dir", output_dir,
        "--num-processes", str(num_processes),
        "--recursive",
    ]

    if use_verbose:
        command.append("--verbose")

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Print output
    if process.returncode != 0:
        print('Command failed. Error:')
        print(error.decode())


def get_result_files(folder_path) -> List[Dict]:
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                _file_path = os.path.join(root, file)
                file_list.append(_file_path)
    return file_list


all_input_files = os.listdir(input_dir)

for input_file in all_input_files:
    file_path = os.path.join(root, input_dir, input_file)
    mime_start = mimetypes.guess_type(file_path)[0]

    if mime_start is not None:
        mime_start = mime_start.split('/')[0]

        if mime_start in ['audio', 'video']:
            print(f"media types : {file_path}")
            result = whisper_model.transcribe(file_path, fp16=False)

            with open(file_path + ".txt", "w", encoding="utf-8") as txt:
                txt.write(result["text"])

process_local(output_dir=output_dir, num_processes=2, input_path=input_dir)
files = get_result_files(output_dir)


def create_local_weaviate_client(db_url: str):
    return weaviate.Client(
        url=db_url,
    )


def get_schema(vectorizer: str = "none"):
    return {
        "classes": [
            {
                "class": "Doc",
                "description": "A generic document class",
                "vectorizer": vectorizer,
                "properties": [
                    {
                        "name": "last_modified",
                        "dataType": ["text"],
                        "description": "Last modified date for the document",
                    },
                    {
                        "name": "document_page_number",
                        "dataType": ["int"],
                        "description": "Text element page number",
                    },
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "Text content for the document",
                    },
                    {
                        "name": "element_parent_id",
                        "dataType": ["text"],
                        "description": "Text element parent id, can be none"
                    },
                    # region file
                    {
                        "name": "document_file_name",
                        "dataType": ["text"],
                        "description": "Document file name",
                    },
                    {
                        "name": "document_file_directory",
                        "dataType": ["text"],
                        "description": "Document file location directory",
                    },
                    {
                        "name": "document_file_type",
                        "dataType": ["text"],
                        "description": "Document file mime type",
                    },
                    {
                        "name": "document_file_path",
                        "dataType": ["text"],
                        "description": "document file path",
                    },
                    {
                        "name": "original_filename",
                        "dataType": ["text"],
                        "description": "original filename",
                    },
                    {
                        "name": "original_file_content",
                        "dataType": ["text"],
                        "description": "original file content",
                    }
                    # endregion
                ],
            },
        ],
    }


def upload_schema(my_schema, weaviate):
    weaviate.schema.delete_all()
    weaviate.schema.create(my_schema)


def count_documents(client: weaviate.Client) -> Dict:
    response = (
        client.query
        .aggregate("Doc")
        .with_meta_count()
        .do()
    )
    count = response
    return count


client = create_local_weaviate_client(db_url=weaviate_url)
my_schema = get_schema()
upload_schema(my_schema, weaviate=client)

embedding_model_path = "model_files/" + embedding_model_name
embedding_model_path = os.path.join(root, embedding_model_path)

embedding_model = SentenceTransformer(embedding_model_name, device=device)
embedding_model.save(
    path=embedding_model_path
)


def compute_embedding(chunk_text: List[str]):
    embeddings = embedding_model.encode(chunk_text, device=device)
    return embeddings


def get_chunks(elements, chunk_under_n_chars=500, chunk_new_after_n_chars=1500):
    for element in elements:
        if not type(element.metadata.data_source) is DataSourceMetadata:
            delattr(element.metadata, "data_source")

        if hasattr(element.metadata, "coordinates"):
            delattr(element.metadata, "coordinates")

    chunks = chunk_by_title(
        elements,
        combine_under_n_chars=chunk_under_n_chars,
        new_after_n_chars=chunk_new_after_n_chars
    )

    for i in range(len(chunks)):
        original_filename = chunks[i].metadata.filename.replace(".json", "")
        file_path = os.path.join(root, input_dir, original_filename)
        original_file_content = open(file_path, "r").read()
        chunks[i] = {
            "last_modified": chunks[i].metadata.last_modified,
            "document_page_number": chunks[i].metadata.page_number,
            "text": chunks[i].text,
            "element_parent_id": chunks[i].metadata.parent_id,
            # region file
            "document_file_name": chunks[i].metadata.filename,
            "document_file_directory": chunks[i].metadata.file_directory,
            "document_file_path": os.path.join(root,
                                               chunks[i].metadata.file_directory + "/" + chunks[i].metadata.filename),
            # "document_file_path": "./" + chunks[i].metadata.file_directory + "/" + chunks[i].metadata.filename,
            "document_file_type": chunks[i].metadata.filetype,
            "original_filename": original_filename,
            "original_file_content": original_file_content
            # endregion
        }

    chunk_texts = [x['text'] for x in chunks]
    embeddings = compute_embedding(chunk_texts)
    return chunks, embeddings


def add_data_to_weaviate(files, client, chunk_under_n_chars=500, chunk_new_after_n_chars=1500):
    for filename in files:
        try:
            elements = partition_json(filename=filename)
            chunks, embeddings = get_chunks(elements, chunk_under_n_chars, chunk_new_after_n_chars)
        except IndexError as e:
            print(e)
            continue
        if use_verbose:
            print(f"Uploading {len(chunks)} chunks for {str(filename)}.")
        for i, chunk in enumerate(chunks):
            if len(files) > 1:
                client.batch.add_data_object(
                    data_object=chunk,
                    class_name="doc",
                    uuid=get_valid_uuid(uuid.uuid4()),
                    vector=embeddings[i],
                )
            else:
                client.data_object.create(
                    data_object=chunk,
                    class_name="doc",
                    uuid=get_valid_uuid(uuid.uuid4()),
                    vector=embeddings[i]
                )
    if len(files) > 1:
        client.batch.flush()


add_data_to_weaviate(
    files=files,
    client=client,
    chunk_under_n_chars=250,
    chunk_new_after_n_chars=500
)
#
# print("count documents with class name 'Doc' : ", count_documents(client=client)['data']['Aggregate']['Doc'])
print("count documents with class name 'Doc' : ", count_documents(client=client))

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 100  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# Make sure the model path is correct for your system!

if use_openai:
    llm = OpenAI(
        api_key=openai_key
    )
else:
    llm = LlamaCpp(
        # llama2 7B
        # model_path="model_files/llama-2-7b-chat.Q4_K_S.gguf",
        # Mistral 7B instruct
        # model_path="model_files/mistral-7b-instruct-v0.2.Q4_K_S.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=use_verbose,  # Verbose is required to pass to the callback manager
    )


def tokenizer(msg, model="text-similarity-davinci-001", show=False):
    encoder = tiktoken.encoding_for_model(model)
    color_pallet = (
        '#FFB347',  # Pastel orange (淡橙色)
        '#FFCC99',  # Peach (桃色)
        '#FFFF66',  # Pastel yellow (淡黃色)
        '#64fa64',  # Tea green (茶綠色)
        '#CCFFCC',  # Aero blue (天青色)
        '#CB99C9',  # Thistle (薊色)
        '#FFB6C1',  # Light pink (淺粉紅色)
        '#FFCCDD',  # Pink lace (粉翠珠色)
        '#CCCCFF'  # Lavender blue (薰衣草藍)
    )
    idx = 0
    tokens = encoder.encode(msg)
    if show:
        for token in tokens:
            print(color(encoder.decode([token]),
                        bg=color_pallet[idx],
                        fg='black'),
                  end='')
            idx = (idx + 1) % len(color_pallet)
        print()
    return tokens


def question_answer(question: str, vectorstore: Weaviate):
    embedding = compute_embedding(question)

    # vector_similar_docs = vectorstore.max_marginal_relevance_search_by_vector(embedding)
    vector_similar_docs = vectorstore.similarity_search_by_vector(embedding)
    # vector_similar_docs = vectorstore.max_marginal_relevance_search(question)

    original_file_content = [x.metadata["original_file_content"] for x in vector_similar_docs]
    original_file_content = list(dict.fromkeys(original_file_content))
    content = [x.page_content for x in vector_similar_docs]

    # get page content

    if use_openai:
        prompt = """
        你是學校秘書，據所有的背景信息，用繁體中文回答問題，當中不要超過150字如果。如果我說'''背景信息未完'''，請不要回答，直至我說'''背景信息完成'''才回答。
    問題:{question}
    背景信息:
    {context}
    
    原文:
    {original_doc}
    
    '''背景信息提供完成'''
        """

        prompt_template = PromptTemplate.from_template(prompt)
    else:
        prompt_template = PromptTemplate.from_template(
            """\
            Given context about the subject, answer the question based on the context provided to the best of your ability.
            Context: {context}
            Question:
            {question}
            Original file content: {original_doc}
            Answer:
            """
        )
    prompt = prompt_template.format(context=content, question=question, original_doc=original_file_content)

    if use_openai:
        messages = [{'role': 'user', 'content': prompt}]

        response = llm.chat.completions.create(
            model='gpt-4-1106-preview',
            # model='gpt-3.5-turbo',
            messages=messages,
            stream=True
        )

        llm_answer = ""

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                llm_answer += content
    else:
        llm_answer = llm(prompt)

    return llm_answer, vector_similar_docs


# region create a vectorstore client

client = weaviate.Client(weaviate_url)

vectorstore = Weaviate(
    client,
    "Doc",
    "text",
    attributes=[
        "last_modified",
        "document_page_number",
        "element_parent_id",
        "document_file_name",
        "document_file_directory",
        "document_file_type",
        "document_file_path",
        "original_filename",
        "original_file_content"
    ]
)

# endregion

# region ask a question to request answer
question = "24班資訊科技綜合津貼"

from opencc import OpenCC

converter = OpenCC('s2t')

answer, similar_docs = question_answer(question, vectorstore)

print("\n\n-------------------------")
print(f"QUERY: {question}")
print("\n-------------------------")
print(f"Answer: {converter.convert(answer)}")
print("\n-------------------------")
for index, result in enumerate(similar_docs):
    print(f"\n-- RESULT {index + 1}:\n")
    print("doc page content : ", {"page_content": result.page_content})
    print("doc metadata : ", {"metadata": result.metadata})

# endregion
