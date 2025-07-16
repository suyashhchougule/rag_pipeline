PROMPT_TMPL = """You are a helpful assistant.
Answer the user’s question **ONLY** with information found in the provided context.
If the answer is not in the context, say "I don’t know".
give the detailed answer based on contet provided.

<context>
{context}
</context>

<question>
{question}
</question>

Respond in JSON with keys:
  "answer": the best answer
  "citations": a list of chunk IDs you used
"""
