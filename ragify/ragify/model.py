import requests
import json

model = "llama3.2"
# model = "phi"


def query_llm(agent_name, prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "agent": agent_name,
            "prompt": prompt,
            "model": model},
        stream=True
    )
    output = ""
    try:
        for line in response.iter_lines():
            if not line.strip():
                continue
            try:
                response_data = json.loads(line)
                if response_data.get("done") is False:
                    output += response_data['response']
            except json.JSONDecodeError as e:
                print("Error decoding line:", line.decode('utf-8'))
                print("JSONDecodeError:", e)
    except requests.exceptions.RequestException as e:
        print("Request error:", e)
    return output


# Define agents
def questioning_prompt(text):
    prompt = (f"Divide the given text {text}, into semantically meaningful chunks and return only the chunks as a list "
              f"of strings separated by newline. Do not include headings, numbering, or any metadata—just the text "
              f"content. Each list element should represent a cohesive semantic unit.")
    return query_llm("agent", prompt)


file_path = 'text.txt' # dummy file for evaluation

# Reading the file and storing content in a string
with open(file_path, 'r', encoding='utf-8') as file:
    text_content = file.read()
    answer = questioning_prompt(text_content)
    print(answer)
