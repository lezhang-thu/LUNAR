import json
from openai import OpenAI


class LLMTemplateMerger:

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-0125",
        api_key: str = None,
        base_url: str = None,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def core(
        self,
        template1: str,
        template2: str,
        llm1,
        llm2,
    ):
        prompt = self._construct_merge_decision_prompt(
            template1,
            template2,
            llm1,
            llm2,
        )
        retry_times = 0
        while retry_times < 3:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role":
                        "system",
                        "content":
                        "You are an expert in log parsing and template extraction. Always respond with valid JSON. "
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0,
                    response_format={"type": "json_object"})
                result = json.loads(response.choices[0].message.content)
                should_merge = result.get("should_merge", False)
                reasoning = result.get("reasoning", "")

                print(f"[LLMMerger] Prompt:\n{prompt}")
                print(f"[LLMMerger] Decision: merge={should_merge}")
                print(f"[LLMMerger] Reasoning: {reasoning}")
                return should_merge
            except Exception as e:
                print(f"[LLMMerger] Error in should_merge:  {e}")
        return None

    def _construct_merge_decision_prompt(
        self,
        template1: str,
        template2: str,
        llm1,
        llm2,
    ) -> str:
        return f"""You are analyzing whether templates 1 & 2 should be merged into template 1.

    **Template 1:** `{template1}`
    Template 1 matches the list of logs: `{str(llm1)}`
    **Template 2:** `{template2}`
    Template 2 matches the list of logs: `{str(llm2)}`

    **Context:**
    - Templates use `<*>` to represent variable parts, matching any seq. of characters
    - We should merge template 2 into template 1 if doing so does not result in significant semantic loss 

    **Response Format (JSON):**
    {{
        "should_merge": true/false,
        "reasoning": "Brief explanation (1-2 sentences)"
    }}
    """
