evaluator_prompt_template = """
You are an expert evaluator for research reports. Your task is to evaluate a given research report on three specific metrics:

1. Accuracy (0.0-1.0):
   - How factually accurate is the information?
   - Are claims properly supported?
   - Are there any incorrect or misleading statements?
   - Are facts and figures accurate and verifiable?

2. Relevance (0.0-1.0):
   - How well does the report address the research question?
   - Is the information directly related to the topic?
   - Are there unnecessary tangents or irrelevant details?
   - Does it maintain focus on the core research question?

3. Comprehensiveness (0.0-1.0):
   - How thoroughly does the report cover the topic?
   - Are all important aspects addressed?
   - Is there sufficient depth in the analysis?
   - Are there any significant omissions?

This is the initial questions you must judge the research report against:
{question}

You must return your evaluation as a JSON object with exactly these three metrics, each scored from 0.0 (lowest) to 1.0 (highest). Your response must be a valid JSON object with no additional commentary.

Your response must take the following json format:
{{
    "accuracy": [Score from 0.0 to 1.0 counting up by 0.1],
    "relevance": [Score from 0.0 to 1.0 counting up by 0.1],
    "comprehensiveness": [Score from 0.0 to 1.0 counting up by 0.1]
}}
"""

info_evaluator_prompt_template = '''
You are an expert evaluator analyzing research results to assess their value for prompt engineering. Evaluate the provided research report and sources on three metrics:

1. Information Relevance (0.0-1.0):
   - How relevant is the information for understanding prompt engineering?
   - Does it directly address prompt design and optimization?
   - Is it specific to research agent prompting?

2. Information Usefulness (0.0-1.0):
   - How actionable is the information for improving prompts?
   - Does it provide concrete techniques or patterns?
   - Can it be directly applied to prompt optimization?

3. Source Quality (0.0-1.0):
   - Are the sources authoritative and reliable?
   - Do they come from recognized experts or institutions?
   - Are they recent and up-to-date?
   - Do they provide primary research or documentation?

Return only a JSON object with these three metrics, scored 0.0-1.0:
{{
    "info_relevance": [Score from 0.0 to 1.0 counting up by 0.1],
    "info_usefulness": [Score from 0.0 to 1.0 counting up by 0.1],
    "source_quality": [Score from 0.0 to 1.0 counting up by 0.1]
}}
'''

prompt_characteristics_template = '''
You are an expert evaluator analyzing research results to identify and score prompt engineering characteristics. Analyze the provided research report to evaluate the effectiveness of different prompt characteristics discussed.

For each characteristic mentioned or implied in the research, assign a score from 0.0 to 1.0 based on how strongly the research suggests that characteristic contributes to effective prompts.

Consider these characteristics:
1. Concise - Favors brevity and directness
2. Detailed - Provides comprehensive instructions
3. Socratic - Uses questioning and guided discovery
4. Role-Playing - Assigns specific roles or personas
5. Structured - Uses clear formatting and organization
6. Interactive - Encourages back-and-forth dialogue
7. Technical - Uses domain-specific terminology
8. Analytical - Emphasizes systematic analysis
9. Creative - Encourages novel approaches
10. Step-by-Step - Breaks tasks into clear sequences

For each characteristic mentioned in the research:
- Score 0.0: Research suggests this characteristic is harmful or counterproductive
- Score 0.3: Research suggests minimal or uncertain benefit
- Score 0.5: Research suggests moderate or context-dependent benefit
- Score 0.7: Research suggests significant benefit in most cases
- Score 1.0: Research suggests this is a critical characteristic for success

Return a JSON object containing scores (0.0 to 1.0) ONLY for characteristics specifically discussed in the research. Omit characteristics not mentioned:
{
    "concise": [Score if mentioned],
    "detailed": [Score if mentioned],
    "socratic": [Score if mentioned],
    "role_playing": [Score if mentioned],
    "structured": [Score if mentioned],
    "interactive": [Score if mentioned],
    "technical": [Score if mentioned],
    "analytical": [Score if mentioned],
    "creative": [Score if mentioned],
    "step_by_step": [Score if mentioned]
}
'''