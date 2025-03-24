from verl import DataProto
import torch
import re
from typing import Dict, List
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

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def update_data_for_critique(data: DataProto, tokenizer, critique_prompt: str = CRITIQUE_PROMPT) -> DataProto:
    # ====================== 输入校验 ======================
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

    # ====================== 核心处理流程 ======================
    def _extract_content(tensor: torch.Tensor, start_tag: str, end_tag: str) -> list:
        decoded = tokenizer.batch_decode(tensor, skip_special_tokens=False)
        print(f"Decoded: {decoded}")
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        return [m.group(1).strip() if (m:=re.search(pattern,t,re.DOTALL)) else "" for t in decoded]
    
    # Step 1: Extract User Queries
    processed_tokens = [
        _pre_process_inputs(pad_token_id, row_ids)
        for row_ids in prompt_ids
    ]
    user_queries = _extract_content(
        processed_tokens, 
        "<|im_start|>user\n", 
        "<|im_end|>"
    )
    print(f"User queries: {user_queries}")
    
    # Step 2: Extract Assistant Responses
    batch_size = response_ids.size(0)
    valid_response_ids_list = []

    # 逐样本处理
    for i in range(batch_size):
        # 计算当前样本的有效response长度
        single_mask = attention_mask[i]  # 当前样本的attention_mask
        valid_length = single_mask[prompt_length:].sum().item()
        
        # 提取有效response tokens
        valid_tokens = response_ids[i, :valid_length]
        valid_response_ids_list.append(valid_tokens)

    # 批量解码（自动处理不等长序列）
    responses_decoded = tokenizer.batch_decode(
        valid_response_ids_list, 
        skip_special_tokens=False,
    )
    print(f"Assistant decoded: {responses_decoded}")
    assistant_responses = [r.split("<|endoftext|>")[0].strip() for r in responses_decoded]

    # ====================== 对齐框架的预处理 ======================
    # Step 3: 严格复用框架预处理逻辑
    input_ids_list, mask_list, pos_ids_list = [], [], []
    
    for q, r in zip(user_queries, assistant_responses):
        # 构造对话结构
        critique_template = copy.deepcopy(critique_prompt)
        formatted_prompt = critique_template.replace(
            "__special_original_question__", q
        ).replace(
            "__special_original_response_", r
        )
        print(formatted_prompt)
        # 应用与训练时一致的模板化处理
        chat_struct = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": formatted_prompt}
        ]
        
        templated_prompt = tokenizer.apply_chat_template(
            chat_struct, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # 复用框架的tokenize函数
        ids, mask = verl_F.tokenize_and_postprocess_data(
            prompt=templated_prompt,
            tokenizer=tokenizer,
            max_length=prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation='right'
        )
        pos_ids = compute_position_id_with_mask(mask)

        input_ids_list.append(ids[0].to(device))  # 保持设备一致性
        mask_list.append(mask[0].to(device))
        pos_ids_list.append(pos_ids[0].to(device))

    # 批量堆叠
    new_input_ids = torch.stack(input_ids_list)
    new_attention_mask = torch.stack(mask_list)
    new_position_ids = torch.stack(pos_ids_list)

    # ====================== 分数处理优化 ======================
    reward_list = []
    for i, response_ids in enumerate(valid_response_ids_list):
        # 获取当前response的长度
        length = len(response_ids)
        last_pos = length - 1
        reward = token_level_scores[i, last_pos].item()  # 假设score是PyTorch Tensor，使用.item()获取标量值
        reward_list.append(reward)

    reward_list = np.array(reward_list, dtype=np.float32)
    assert all(r in {-1, -0.5, 1} for r in reward_list), f"Invalid reward found: {reward_list}"
    
    # ====================== 构建新数据 ======================
    new_batch = TensorDict({
        'input_ids': new_input_ids,
        'attention_mask': new_attention_mask,
        'position_ids': new_position_ids
    }, batch_size=batch_size)

    new_non_tensor = {
        k: np.copy(v) if isinstance(v, np.ndarray) else v.copy()
        for k, v in data.non_tensor_batch.items()
    }

    # Format the reward_list values to strings with one decimal place
    formatted_ground_truths = np.char.mod("%.1f", reward_list)

    # Create a list of dictionaries with formatted 'ground_truth' and original 'style'
    reward_dicts = [
        {"ground_truth": gt, "style": d["style"]}
        for gt, d in zip(formatted_ground_truths, data.non_tensor_batch["reward_model"])
    ]

    # Convert the list of dictionaries to a NumPy array of objects (dictionaries)
    new_non_tensor["reward_model"] = np.array(reward_dicts, dtype=object)
    
    # operate on object np.array
    new_index = data.non_tensor_batch['index'].copy()
    new_index[:] = [x*2 for x in new_index]  # 元素级操作
    new_non_tensor['index'] = new_index

    # ====================== 清理字段 ======================
    for k in ['prompts', 'responses', 'token_level_scores']:
        new_batch.pop(k, None)
    data.non_tensor_batch.pop('uid', None)

    return DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor,
    )
