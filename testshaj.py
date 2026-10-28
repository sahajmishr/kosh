from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="i am not gonna put my api here .. if you wanna use this dummy test use yours man ....")
print(c.chat.completions.create(model="qwen-3.8-27b",
      messages=[{"role":"user","content":"hi"}]).choices[0].message.content)