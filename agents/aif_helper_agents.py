from prompts.aif_prompts import evaluator_prompt_template, info_evaluator_prompt_template
from models.openai_models import get_open_ai_json
import json

class EvaluatorAgent:
    """
    Agent for evaluating research report quality on accuracy, relevance, and comprehensiveness.
    """
    def __init__(self, model=None, server=None, temperature=0):
        self.model = model or "gpt-4o"
        self.server = server or "openai"
        self.temperature = temperature
        
    def invoke(self, research_report, research_question):
        """
        Evaluate a research report and return numerical scores.
        
        Args:
            research_report (str): The final research report to evaluate
            
        Returns:
            dict: Dictionary containing scores for accuracy, relevance, and comprehensiveness
        """

        if not research_report:
            print("Warning: Empty research report received")
            return {
                'accuracy': 0.0,
                'relevance': 0.0,
                'comprehensiveness': 0.0
            }

        evaluator_prompt = evaluator_prompt_template.format(
            question=research_question,
        )
        
        messages = [
            {"role": "system", "content": evaluator_prompt},
            {"role": "user", "content": f"This is the research report you must evaluate: {research_report}"}
        ]
        
        try:
            llm = get_open_ai_json(model=self.model, temperature=self.temperature)
            ai_msg = llm.invoke(messages)
            scores = json.loads(ai_msg.content)
            
            # Ensure scores are within 0-1 range
            for key in scores:
                scores[key] = max(0.0, min(1.0, float(scores[key])))
                
            return scores
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return {
                'accuracy': 0.5,
                'relevance': 0.5,
                'comprehensiveness': 0.5
            }
        

class InfoEvaluatorAgent:
    def __init__(self, model=None, server=None, temperature=0):
        self.model = model or "gpt-4o"
        self.server = server or "openai"
        self.temperature = temperature
        
    def invoke(self, research_report, selected_sources):
        if not research_report:
            return {
                'info_relevance': 0.0,
                'info_usefulness': 0.0,
                'source_quality': 0.0
            }

        messages = [
            {"role": "system", "content": info_evaluator_prompt_template},
            {"role": "user", "content": f"Research Report:\n{research_report}\n\nSelected Sources:\n{selected_sources}"}
        ]
        
        try:
            llm = get_open_ai_json(model=self.model, temperature=self.temperature)
            ai_msg = llm.invoke(messages)
            scores = json.loads(ai_msg.content)
            
            for key in scores:
                scores[key] = max(0.0, min(1.0, float(scores[key])))
                
            return scores
            
        except Exception as e:
            print(f"Information evaluation error: {str(e)}")
            return {
                'info_relevance': 0.5,
                'info_usefulness': 0.5,
                'source_quality': 0.5
            }