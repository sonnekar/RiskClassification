import multiprocessing
import ollama

import time

def chat_worker(model, messages):
    return ollama.chat(model=model, messages=messages)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=16) 
    
    t_s = time.time()
    inputs = [
        ("llama2:latest", [{"role": "user", "content": "Tell me a joke."}]),
        ("llama2:latest", [{"role": "user", "content": "What is the capital of France?"}]),
    ] * 100
    
    results = pool.starmap(chat_worker, inputs)
    print(results) 
    print(f'time w/ multiprocessing: {time.time() - t_s}')

    t_s = time.time()
    for input in inputs:
        output = ollama.chat(
            model    = 'llama2:latest', 
            messages = input[1],
            #options  = {'temperature': 0, 'num_predict': 1}
        )
    print(f'time w/out multiprocessing: {time.time() - t_s}')
