# Argument-Value-Extractor
Streamlit app that, when given argumentative sentence, outputs underlying primary value motivating that sentence. Compares the outputs of a fine-tuned BERT model on a dataset of ~300 sentences with GPT 4o that utilizes a value ontology to ground its predictions. The values the models are choosing from are the following: Autonomy, Fairness, Life, Quality of Life, Safety, Economic Growth and Preservation, Responsibility, Honesty, Innovation, Sustainability. The purpose of this tool is to eventually create a full-fleged debate agent that can assess the values of an argument, the measures used for that value, and how well the value is achieved with a proposed solution. The first steps to doing this include the extraction of the value and the measures being used to quantify that value. 

# Running
You can access the app through the following URL: argument-value-extractor-mz.streamlit.app

# GPT Prompting and Retrieval
The prompt is built using the openAI_prompt.py file. The sentence is first passed to retrieval.py, which uses a word embedding model to measure the cosine similarity between the sentence and the values as defined in the ontology. It outputs the top 3 candidates as well as their measures. This is passed into GPT, along with the original sentence. More details can be seen in the prompting file.

# BERT: Fine tuning and Dataset
The BERT model used to fine-tune on was the base BERT model, on HF as "bert-base-uncased". The dataset used to fine tune contains 100 random sentences from the "Argument Based Aspect Mining" dataset found here on Kaggle: https://www.kaggle.com/datasets/trtmio/aspect-based-argument-mining. These sentences were then manually read and annotated with a corresponding value, and are identified with their original id number. Note that the manually annotated sentences include measures, though this is not used in the fine-tuning. The remainder of the 200 sentences were generated using GPT5 then proof read, and are labelled in the dataset as SYN-(###). The specific fine tuning script can be found in the BERT_finetune folder. After fine-tuning, the F1 score improved from around 68 to 89. The model was uploaded to HF under "mzq34/bert-values-classifier", and this is accessed by the Streamlit app.

# Next Steps:
I am currently working towards a multinomial classifier to assess the polarity of each value that is extracted. The primary challenge of this approach is the data required to train the classifier. Once this is complete, I plan to 
