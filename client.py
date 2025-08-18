import requests

API_URL = "http://localhost:8000/ask"


def ask_question(question: str):
    payload = {
        "question": question,
        "top_k": 3
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # 如果请求失败会抛出HTTPError
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求API时报错: {e}")
        return None


if __name__ == '__main__':
    while True:
        question = input("请输入你的问题(exit退出): ").strip()
        if question.lower() in ("exit", "quit"):
            break
        response = ask_question(question)
        if response:
            print(response['answer'])