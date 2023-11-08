from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


question_prompt_template = \
"""Use a seguinte parte de um documento extenso para verificar se algum texto é relevante para responder à pergunta.
Retorne qualquer texto relevante de forma literal.
{context}
Pergunta: {question}
Texto relevante, se houver algum:"""


qa_prompt = PromptTemplate(
    template=question_prompt_template,
    input_variables=["question"],
)

