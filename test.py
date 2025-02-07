import asyncio
from vllm import AsyncLLM, SamplingParams

async def generate_text():
    llm = AsyncLLM(model="TheBloke/Qwen2.5-14B-Chat-GPTQ")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    prompts = ["Tell me a joke.", "What is reinforcement learning?", "Describe a black hole."]
    
    results = await asyncio.gather(*[llm.generate(prompt, sampling_params) for prompt in prompts])
    
    for output in results:
        print(output)

asyncio.run(generate_text())
