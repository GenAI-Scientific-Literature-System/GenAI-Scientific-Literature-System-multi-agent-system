import json
import re

def parse_output(text):
    try:
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return{
            "reliability_score": 0.5,
            "confidence": "Medium",
            "justification" : ["Fallback due to parsing error"]
        }