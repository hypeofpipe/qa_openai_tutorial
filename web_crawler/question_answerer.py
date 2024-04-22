import numpy as np
import pandas as pd
from openai import OpenAI
import os
import tiktoken

from scipy import spatial

from utils.embedding_utils import distances_from_embeddings

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

EMBEDDING_MODEL = 'text-embedding-3-small'
GPT_MODEL = 'gpt-3.5-turbo'

class QuestionAnswererV1:
    def __init__(self):
        df=pd.read_csv('processed/embeddings_v2.csv', index_col=0)
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        
        self.df = df
    def create_context(
        self, question, max_len=3600
    ):
        """
        Create a context for a question by finding the most similar context from the dataframe
        """

        df = self.df
        
        # Get the embeddings for the question
        q_embeddings = client.embeddings.create(input=question, model=EMBEDDING_MODEL).data[0].embedding

        # Get the distances from the embeddings
        df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

        added_contexts = set()  # Keep track of added contexts to prevent duplication
        cur_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for i, row in df.sort_values('distances', ascending=True).iterrows():
            # Check if the context has already been added, skip if duplicate
            if row["text"] in added_contexts:
                continue

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            # Else add it to the text that is being returned
            added_contexts.add(row["text"])

        # Return the context
        return "\n\n###\n\n".join(added_contexts)
    def ask(
        self,
        model="gpt-3.5-turbo",
        question = "Як проходить день?",
        max_len=6200,
        debug=False,
        max_tokens=1000,
        stop_sequence=None,
    ):
        """
        Answer a question based on the most similar context from the dataframe texts
        """
        context = self.create_context(
         question, max_len=max_len,
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
                    {"role": "system", "content": "Дай відповідь, беручи під увагу $CONTEXT, який буде поданий нижче. Якщо на підставі контексту, ти не можеш надати відповідь - скажи \"Не можу відповісти на це питання.\""},
                    {"role": "user", "content": f"$CONTEXT: {context}\n\n---\n\nПитання: {question}\nВідповідь:"}
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
    
class QuestionAnswererV2:
    def __init__(self):
        df=pd.read_csv('processed/embeddings_v2.csv', index_col=0)
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        
        self.df = df
    # search function
    def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]
    
    def num_tokens(text: str, model: str = GPT_MODEL) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    def create_context(
        self,
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
    ) -> str:
        strings = self.strings_ranked_by_relatedness(query, df)
        introduction = 'Use the below articles on law 3261 to answer the subsequent question. If the answer cannot be found in the data, write "Я не знайшов відповіді в матеріалах, до яких я маю доступ."'
        question = f"\n\nQuestion: {query}"
        message = introduction
        for string in strings:
            next_section = f'\n\nRelated data:\n"""\n{string}\n"""'
            if (
                self.num_tokens(message + next_section + question, model=model)
                > token_budget
            ):
                break
            else:
                message += next_section
        return message + question
    
    def ask(
        self,
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        message = self.query_message(query, df, model=model, token_budget=token_budget)
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": "Ти відповідаєш на питання відносно законопроекту про мобілізацію."},
            {"role": "user", "content": message},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        response_message = response.choices[0].message.content
        return response_message

if __name__ == "__main__":
    questionAnswererV1 = QuestionAnswererV1()
    questionAnswererV2 = QuestionAnswererV2()

# V1.1 (used text-embedding-3-small):
    # print(questionAnswererV1.ask(question="Як проходить день?", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="Які зміни передбачає законопроект про мобілізацію?", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="Які зміни передбачає законопроект про мобілізацію?", model="gpt-4"))
        # response "На підставі наданого контексту, не можу відповісти на це питання."
    # print(questionAnswererV1.ask(question="Які зміни передбачає законопроект про мобілізацію?", model="gpt-4-turbo"))
        # response "Не можу відповісти на це питання."
    # print(questionAnswererV1.ask(question="Якщо я чоловік 23 років, чи по закону я зобовязаний служити?", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="Чи можуть мені бути надані консульскі послуги?", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="У мене три автівки, чи законопроект передбачає їх мобілізацію?", debug=True))
        # response "Я не знаю"

    # print(questionAnswererV1.ask(question="Скільки заробляє військовослужбовець?", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="Скільки заробляє військовослужбовець?", model="gpt-4", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="Який оклад військовослужбовця?", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="Які вимоги до військовослужбовця?", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="Коли військовослужбовцю дається відпустка?", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="Чи може військовослужбовець відмовитися від служби?", debug=True))
        # response "Я не знаю"
    # print(questionAnswererV1.ask(question="Коли військовослужбовець звільняється від виконання службових обовязків?", debug=True))
        # response "Коли військовослужбовець звільняється від виконання службових обов'язків до набуття права на пенсію за вислугу років, крім випадків, передбачених законом."
    # print(questionAnswererV1.ask(question="Коли військовослужбовець звільняється від виконання службових обовязків?", model="gpt-4", debug=True))
        # response "Військовослужбовець звільняється від виконання службових обов'язків у разі необхідності догляду за хворою дитиною віком до 14 років, яка потребує стаціонарного лікування. Звільнення від виконання службових обов'язків здійснюється із збереженням грошового забезпечення на весь період догляду за хворою дитиною."

