from typing import Any, Optional

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import Approach, ThoughtStep
from core.authentication import AuthenticationHelper


class RetrieveThenReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the AI Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = ("""
        Microsoft DC + Sustainability v2
        You are an expert on Microsoft views and documents dealing with technology policy in countries around the world.
        Your task is to think carefully about what the user is asking for and then retrieve and analyze the relevant information needed to answer the user’s question in the most comprehensive, the most fully detailed, and the most accurate manner possible.
        BEFORE creating your answer, make CERTAIN you understand the specific type and format of answer the user is seeking. Is the user asking a simple factual question? Are they asking for a list of items in a certain category? Are they asking for a set of talking points or the content for a slide deck? Are they asking for some other type of information or specific format?
        If you are uncertain whether the information you have is relevant to the user’s question, DO make your best effort to use that information to construct an appropriate answer, while obeying all of the rules I give you below.
        If you don’t find an answer that is plausibly derived from or supported by your source documents, DO NOT for any reason use prior knowledge to answer the question.
        Once you have understood the type and format of response that the user is seeking, DO construct your answer STEP BY STEP in the user’s desired format, making certain that you have given the user ALL relevant information and have not omitted any relevant item, fact, number, data point, or name.
        DO NOT BE LAZY in your answers. You always give the user ALL of the relevant information and items in your source documents.
        If the user asks you for Talking Points, DO give each point a concise headline followed by a meaty and detailed paragraph of text.
        If the user asks for the content for a slide deck or presentation, DO give each slide a concise title that clearly expresses the main point of the slide followed by four or five substantial bullet points that convey the relevant content as described below. Every slide should also have a very detailed and meaty speaker note which provides a comprehensive script for the speaker presenting this slide. The language of the note should be unpretentious and free of marketing hype.
        If it seems like the user is asking for a list of items in a certain category, you MUST give ALL members of the list. DO NOT for any reasons abbreviate the list. DO NOT FAIL to recognize when the user is asking for a list, even if they don’t use that word.
        DO NOT write short answers. You MUST always write meaty, multi-paragraph answers that treat the subject comprehensively. As part of these answers, you may use bullet points to summarize lists of items when this will help the user absorb the information you present.
        DO pay close attention to any quantitative data points or financial numbers in your source documents, and answer any questions about these data points accurately.
        DO NOT force conclusions or inferences that are not supported by your source documents or by your knowledge of the world.
        DO NOT make any statement or cite any fact not supported by your source documents.
        DO NOT answer questions that are not related to your source documents.
        DO write in the business professional style of an article in Harvard Business Review without hype or pretentious language.
        When you are asked about countries, look for ALL countries correspond to the user’s question.
        DO NOT BE LAZY!
        
        + "Answer the following question using only the data provided in the sources below. "
        + "Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. "
        + "If you cannot answer using the sources below, say you don't know. Use below example to answer"
    """)

    # shots/sample conversation
    question = """
    'How much is Microsoft investing in data centers in France and Italy?'

    Sources:
    info1.txt: Microsoft announced in May 2024 that it will 4 billion euros in France for cloud and AI infrastructure, AI skilling, and French Tech acceleration.
    info2.pdf: Microsoft is investing €4.3B to boost AI infrastructure and cloud capacity in Italy.
    info3.pdf: Microsoft announced in February 2024 that it will invest 3.2 billion euros in Germany in data centers for cloud and AI applications and train more than 1.2 million people in digital skills by the end of 2025.
    info4.pdf: As part of its commitment to promote digital innovation and the responsible use of Artificial Intelligence that benefits Spanish companies and publicadministrations, Microsoft plans to quadruple its investments in AI and Cloud infrastructure in Spain during 2024-2025, to reach $2.1 billion dollars.
"""
    answer = "Microsoft will invest 4 billions euros in data centers for France and 4.3 billion euros for data centers in Italy. [info1.txt] [info2.pdf]"

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model, self.ALLOW_NON_GPT_MODELS)

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        q = messages[-1]["content"]
        if not isinstance(q, str):
            raise ValueError("The most recent message content must be a string.")
        overrides = context.get("overrides", {})
        seed = overrides.get("seed", None)
        auth_claims = context.get("auth_claims", {})
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(q))

        results = await self.search(
            top,
            q,
            filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        # Process results
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Append user message
        content = "\n".join(sources_content)
        user_content = q + "\n" + f"Sources:\n {content}"

        response_token_limit = 1024
        updated_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=overrides.get("prompt_template", self.system_chat_template),
            few_shots=[{"role": "user", "content": self.question}, {"role": "assistant", "content": self.answer}],
            new_user_content=user_content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
            fallback_to_default=self.ALLOW_NON_GPT_MODELS,
        )

        chat_completion = await self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=updated_messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            seed=seed,
        )

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Search using user query",
                    q,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    updated_messages,
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        return {
            "message": {
                "content": chat_completion.choices[0].message.content,
                "role": chat_completion.choices[0].message.role,
            },
            "context": extra_info,
            "session_state": session_state,
        }
