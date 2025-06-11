from verl import DataProto
import torch
import re
from typing import Dict, List, Optional
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import numpy as np
import copy
from tensordict import TensorDict

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

QWEN_MATH_SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""

USER_START_TAG = "<|im_start|>user\n"
USER_END_TAG = "<|im_end|>"
# ASSISTANT_START_TAG = "<|im_start|>assistant\n"
# Because we add generation prompt at the end of the prompt, we do not need to add assistant start tag
ASSISTANT_END_TAG = "<|endoftext|>"


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def extract_content_in_between(decoded_text: List[str], start_tag: Optional[str] = None, end_tag: Optional[str] = None) -> List[str]:
    # normalise “missing tag” → None  (so we can test with `is None`)
    start_tag = start_tag or None
    end_tag   = end_tag   or None

    # Build the pattern according to which tags are present
    if start_tag and end_tag:
        pattern = (re.escape(start_tag) + r"(.*?)" + re.escape(end_tag))
    elif start_tag and not end_tag:
        pattern = re.escape(start_tag) + r"(.*)$"
    elif not start_tag and end_tag:
        pattern = r"^(.*?)" + re.escape(end_tag)
    else: # no tags at all → whole string
        return [t.strip() for t in decoded_text]
    # Apply the search per string (first match only)
    return [
        (m.group(1).strip() if (m := re.search(pattern, t, re.DOTALL)) else "")
        for t in decoded_text
    ]

def update_data_for_critique(data: DataProto, tokenizer, critique_prompt_idx: int = 0, critique_system_prompt: str = QWEN_MATH_SYSTEM_PROMPT) -> DataProto:
    critique_prompt = CRITIQUE_PROMPT_POOL[critique_prompt_idx]
    required_batch_keys = {'input_ids', 'responses', 'attention_mask', 'prompts', 'token_level_scores'}
    if not all(k in data.batch for k in required_batch_keys):
        raise ValueError("Invalid DataProto structure")
    
    prompt_ids = data.batch['prompts'] # Note that after generation, the prompt field stores the question's input_ids
    # input_ids = data.batch['input_ids'] # The input_ids field stores the whole sequence_ids
    response_ids = data.batch['responses']
    attention_mask = data.batch['attention_mask']
    token_level_scores = data.batch['token_level_scores']

    batch_size = data.batch['input_ids'].size(0)
    device = data.batch['input_ids'].device
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prompt_length = prompt_ids.shape[-1] # max length of the prompt (input_ids)
    
    # ====================== 1/4 Extract User & Assistant Content ======================
    # 1. Extract User Queries
    processed_prompts = [
        _pre_process_inputs(pad_token_id, row_ids)
        for row_ids in prompt_ids
    ]
    
    decoded_prompts = tokenizer.batch_decode(
        processed_prompts,
        skip_special_tokens=False
    )
    user_queries = extract_content_in_between(decoded_prompts, USER_START_TAG, USER_END_TAG)
    # print(f"User queries: {user_queries}")
    
    # 2. Extract Assistant Responses
    batch_size = response_ids.size(0)
    valid_response_ids_list = []
    
    # process logic based on reward_manager naive.py _call_reward()
    for i in range(batch_size):
        single_mask = attention_mask[i]
        valid_length = single_mask[prompt_length:].sum().item()
        valid_tokens = response_ids[i, :valid_length]
        valid_response_ids_list.append(valid_tokens)

    decoded_responses = tokenizer.batch_decode(
        valid_response_ids_list, 
        skip_special_tokens=False,
    )
    # Again, we start from the beginning as the assistant's start_tag is included in the prompt not generation
    assistant_responses = extract_content_in_between(decoded_responses, start_tag=None, end_tag=ASSISTANT_END_TAG)
    # print(f"Assistant responses: {assistant_responses[:2]}")

    # ====================== 2/4 Construct Critique Prompts ======================
    input_ids_list, mask_list, pos_ids_list = [], [], []
    
    for q, r in zip(user_queries, assistant_responses):
        critique_template = copy.deepcopy(critique_prompt)
        formatted_prompt = critique_template.replace(
            "__special_original_question__", q
        ).replace(
            "__special_original_response__", r
        )
        # print(formatted_prompt + '\n')
        chat_struct = [
            {"role": "system", "content": critique_system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        templated_prompt = tokenizer.apply_chat_template(
            chat_struct, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # process logic based on utils/dataset/rl_dataset.py RLHFDataset __getitem__
        ids, mask = verl_F.tokenize_and_postprocess_data(
            prompt=templated_prompt,
            tokenizer=tokenizer,
            max_length=prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation='right'
        )
        pos_ids = compute_position_id_with_mask(mask)

        input_ids_list.append(ids[0].to(device))
        mask_list.append(mask[0].to(device))
        pos_ids_list.append(pos_ids[0].to(device))

    new_input_ids = torch.stack(input_ids_list)
    new_attention_mask = torch.stack(mask_list)
    new_position_ids = torch.stack(pos_ids_list)

    # ====================== 3/4 Extract Reward From Reward Tensor ======================
    reward_list = []
    for i, response_ids in enumerate(valid_response_ids_list):
        length = len(response_ids)
        last_pos = length - 1
        reward = token_level_scores[i, last_pos].item()
        reward_list.append(reward)

    reward_list = np.array(reward_list, dtype=np.float32)
    assert all(r in {-1, -0.5, 1} for r in reward_list), f"Invalid reward found: {reward_list}"
    
    # ====================== 4/4 Construct DataProto and Clean Keys ======================
    new_batch = TensorDict({
        'input_ids': new_input_ids,
        'attention_mask': new_attention_mask,
        'position_ids': new_position_ids
    }, batch_size=batch_size)

    new_non_tensor = {
        k: np.copy(v) if isinstance(v, np.ndarray) else v.copy()
        for k, v in data.non_tensor_batch.items()
    }

    formatted_ground_truths = np.char.mod("%.1f", reward_list)

    reward_dicts = [
        {"ground_truth": gt, "style": d["style"]}
        for gt, d in zip(formatted_ground_truths, data.non_tensor_batch["reward_model"])
    ]
    
    new_non_tensor["reward_model"] = np.array(reward_dicts, dtype=object)
    
    # operate on object-based np.array
    new_index = data.non_tensor_batch['index'].copy()
    new_index[:] = [x*2 for x in new_index]
    new_non_tensor['index'] = new_index

    for k in ['prompts', 'responses', 'token_level_scores']:
        new_batch.pop(k, None)
    data.non_tensor_batch.pop('uid', None)

    return DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor,
    )
