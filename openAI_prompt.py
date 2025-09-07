from openai import OpenAI
import openai
import torch 
import json, re
import retrieval


def build_prompt(candidates, argument):
    prompt = "You are an information extraction model. " \
    "You must determine the primary underlying value of an argument.\n " 
    "Use the measures provided as a way to quantify a value " \
    "These are the only values you may use. Choose the value that alligns closest with the argument. Each line is one value and its corresponding measure:"

    #only inquire about the given candidates
    for c in candidates:
        name = c["name"]
        measures = c["measures_topk"]

        prompt += f"\n VALUE NAME: {name} | VALUE MEASURES: {measures} \n"

    arg = f"This is the argument you must assess: {argument} \n"
    prompt += arg
    schema = (
        "Return ONLY the value text. No punctuation, no extra words."
    )

    return prompt + schema 


def prompt_gpt(argument):
    candidates = retrieval.run_retrival(argument)
    return build_prompt(candidates, argument)
    

