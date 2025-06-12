
USER_START_TAG = "<|im_start|>user\n"
USER_END_TAG = "<|im_end|>"
ASSISTANT_START_TAG = "</think>\n"
# Because we add generation prompt at the end of the prompt, we do not need to add assistant start tag
ASSISTANT_END_TAG = "<|im_end|>"


QWEN_MATH_SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""

CRITIQUE_PROMPT = """Below you are presented with a question and a tentative response. Your task is to evaluate and assign a rating to the response based on the following clear criteria:

Rating Criteria:

1. Missing final answer enclosed in \\boxed{} at the end: assign \\boxed{-1}.
2. Correct response with the final answer enclosed in \\boxed{} at the end: assign \\boxed{1}.
3. Incorrect response with the final answer enclosed in \\boxed{} at the end: assign \\boxed{-0.5}.

### Question Begin ###
__special_original_question__
### Question End ###

### Response Begin ###
__special_original_response__
### Response End ###

Briefly summarize your analysis, then clearly state your final rating value enclosed in \\boxed{} at the end.
"""

CRITIQUE_PROMPT_EXT = """Below you are presented with a question and a tentative response. Your task is to evaluate and assign a rating to the response based on the following clear criteria:

Rating Criteria:

1. Missing final answer enclosed in \\boxed{} at the end: assign \\boxed{-1}.
2. Correct response with the final answer enclosed in \\boxed{} at the end: assign \\boxed{1}.
3. Incorrect response with the final answer enclosed in \\boxed{} at the end: assign \\boxed{-0.5}.

### Question Begin ###
__special_original_question__
### Question End ###

### Response Begin ###
__special_original_response__
### Response End ###

Thoroughly analyze the response, identify any reasoning errors or format mismatches, and conclude with your final rating enclosed in \\boxed{} at the end.
"""

CRITIQUE_PROMPT_2 = """Below are a question and a tentative solution. Think step by step to verify the answer and assign a final rating to the solution based on the following criteria:

1. Missing final answer enclosed in \\boxed{} at the end: assign \\boxed{-1}.
2. Correct response with the final answer enclosed in \\boxed{} at the end: assign \\boxed{1}.
3. Incorrect response with the final answer enclosed in \\boxed{} at the end: assign \\boxed{-0.5}.
    
### Question ###
__special_original_question__

### Response ###
__special_original_response__
"""

CRITIQUE_PROMPT_3 = """Below are a question and a tentative solution. Think step by step to verify the correctness of the solution and assign a final rating at last. If the solution is correct, assign \\boxed{1}. If the solution is incorrect, assign \\boxed{-0.5}. If the solution is missing a final answer enclosed in \\boxed{} at the end, assign \\boxed{-1}.
    
### Question ###
__special_original_question__

### Solution ###
__special_original_response__
"""

CRITIQUE_PROMPT_POOL = [
    CRITIQUE_PROMPT,
    CRITIQUE_PROMPT_2,
    CRITIQUE_PROMPT_3,
    CRITIQUE_PROMPT_EXT,
]
