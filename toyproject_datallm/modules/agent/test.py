# # prompt.py 파일에서 agent_prompt_en 가져오기
# from prompt_templates.prompt import Prompt


# def main():
#     # test 클래스의 인스턴스 생성

#     prompt_instance = Prompt()
#     agent_prompt = prompt_instance.get_prompt("ko")
#     print(agent_prompt)
# if __name__ == "__main__":
#     main()


import requests

headers = {
    "Content-Type": "application/json",
}

data = {
    'inputs': 'What is Deep Learning?',
    'parameters': {
        'max_new_tokens': 20,
    },
}

response = requests.post('http://tgi_test_yw2:8006/generate', headers=headers, json=data)
print(response.json())