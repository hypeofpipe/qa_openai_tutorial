import numpy as np
import pandas as pd
from openai import OpenAI
import os

from utils.embedding_utils import distances_from_embeddings

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Дай відповідь, беручи під увагу $CONTEXT, який буде поданий нижче. Якщо на підставі $CONTEXT не можна видати відповідь, скажи \"Я не знаю\"\n\n"},
                {"role": "user", f"content": "$CONTEXT: {context}\n\n---\n\nПитання: {question}\nВідповідь:"}
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""
    

# print(answer_question(df, question="Як проходить день?", debug=True))
# response "Я не знаю"
# print(answer_question(df, question="Які зміни передбачає законопроект про мобілізацію?", debug=True))
# response "Я не знаю"
# print(answer_question(df, question="Якщо я чоловік 23 років, чи по закону я зобовязаний служити?", debug=True))
# response "Я не знаю"
# print(answer_question(df, question="Чи можуть мені бути надані консульскі послуги?", debug=True))
# response "Я не знаю"
# print(answer_question(df, question="У мене три автівки, чи законопроект передбачає їх мобілізацію?", debug=True))
# response "Я не знаю"

# print(answer_question(df, question="Скільки заробляє військовослужбовець?", debug=True))
# response "Я не знаю"

# print(answer_question(df, question="Який оклад військовослужбовця?", debug=True))
# response "Я не знаю"

# print(answer_question(df, question="Які вимоги до військовослужбовця?", debug=True))
# response "Я не знаю"

# print(answer_question(df, question="Коли військовослужбовцю дається відпустка?", debug=True))

