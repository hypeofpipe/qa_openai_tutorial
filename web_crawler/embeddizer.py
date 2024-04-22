from openai import OpenAI
import pandas as pd
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

df = pd.read_csv('processed/shortened.csv', index_col=0)

# df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input=x, model='text-embedding-ada-002').data[0].embedding)
df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input=x, model='text-embedding-3-small').data[0].embedding)

df.to_csv('processed/embeddings_v2.csv')
df.head()