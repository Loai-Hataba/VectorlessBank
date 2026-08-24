"""
pipeline/prompt_templates.py

WHAT THIS FILE DOES
--------------------
Holds every piece of TEXT we send to the LLM: the system prompt
(instructions/persona) and the user prompt template (how we present
retrieved context + the question). No logic lives here, only strings
and simple string-formatting functions.

WHY IT EXISTS AS ITS OWN FILE
--------------------------------
Prompt wording is something you'll tune constantly while testing --
tightening instructions, adding guardrail language, adjusting tone.
Keeping it separate from rag_pipeline.py means you (or your partner)
can edit HOW the bot talks without touching the code that decides
WHAT data it sees. It also makes it trivial to compare two prompt
versions side by side later, or A/B test them in the evaluation
pipeline (Phase 2).

INPUTS  : retrieved context (str), conversation/question (str)
OUTPUTS : ready-to-send prompt strings
"""

SYSTEM_PROMPT = """You are a helpful assistant for Banque Misr's retail banking \
customers. You answer questions about credit cards, merchant offers \
(installments and discounts), and ongoing campaigns.

Rules you must follow:
1. Base your answer ONLY on the "Retrieved information" provided below, \
except for general banking knowledge questions (e.g. "what is APR", \
"how does a credit card work") where you may use your own knowledge.
2. If the retrieved information does not contain the answer, say so \
clearly instead of guessing or making up numbers, fees, or dates.
3. Be concise and direct. Use bullet points for lists of terms/fees.
4. Never invent a card name, merchant name, fee, or date that is not \
present in the retrieved information.
"""


def build_user_prompt(
    question: str,
    context_text: str,
    data_was_insufficient: bool = False,
) -> str:
    """
    Combine the retrieved context and the user's question into the
    final prompt sent to the LLM.

    If context_text is empty (no source matched anything), we still
    send the question but flag that nothing was retrieved, so the
    LLM leans on rule #1 above (general knowledge only, or say it
    doesn't have specific info).

    data_was_insufficient is set by Corrective RAG when grading -- and
    then one widened retry -- still failed to turn up a record that
    genuinely answers the question. There is an important difference
    between "here is the context" and "here is the closest we found,
    and it probably is not enough": without being told, the model
    treats a weak partial match as though it were the answer and writes
    confidently from it. Saying so explicitly is what lets it decline
    instead, which is the entire point of the corrective branch.
    """
    if context_text.strip():

        if data_was_insufficient:
            return (
                f"Retrieved information (INCOMPLETE -- a wider search "
                f"still found nothing that directly answers this "
                f"question; what follows is only the closest match and "
                f"may not contain the answer):\n"
                f"{context_text}\n\n"
                f"---\n"
                f"Customer question: {question}\n\n"
                f"If the information above does not actually answer the "
                f"question, say so plainly. Do not fill the gap with "
                f"figures, names or terms that are not shown above."
            )

        return (
            f"Retrieved information:\n"
            f"{context_text}\n\n"
            f"---\n"
            f"Customer question: {question}"
        )
    else:
        return (
            f"Retrieved information: (nothing matched this question in our "
            f"cards, offers, or campaigns data)\n\n"
            f"---\n"
            f"Customer question: {question}"
        )
