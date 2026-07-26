import torch
from PIL import Image

from attack_core.u2bench import resolve_label


def build_chat_prompt(proc, system_prompt: str, question_text: str, reasoning_off: bool = False):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question_text},
                {"type": "image"},
            ],
        }
    )
    if reasoning_off:
        try:
            prompt_text = proc.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = proc.apply_chat_template(messages, add_generation_prompt=True)
    else:
        prompt_text = proc.apply_chat_template(messages, add_generation_prompt=True)
    return prompt_text


def _prepare_vlm_inputs(
    proc,
    image: Image.Image,
    system_prompt: str,
    question_text: str,
    reasoning_off: bool = False,
):
    prompt_text = build_chat_prompt(proc, system_prompt, question_text, reasoning_off=reasoning_off)
    model_inputs = proc(text=prompt_text, images=[image], return_tensors="pt")
    return prompt_text, model_inputs


@torch.no_grad()
def score_candidate(
    vlm,
    proc,
    image: Image.Image,
    system_prompt: str,
    question_text: str,
    candidate: str,
    reasoning_off: bool = False,
) -> float:
    base_prompt, prompt_inputs = _prepare_vlm_inputs(
        proc,
        image,
        system_prompt,
        question_text,
        reasoning_off=reasoning_off,
    )
    full_inputs = proc(text=base_prompt + " " + candidate, images=[image], return_tensors="pt")

    device = next(vlm.parameters()).device
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

    prompt_length = prompt_inputs["input_ids"].shape[1]
    input_ids = full_inputs["input_ids"]
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100

    outputs = vlm(
        input_ids=input_ids,
        attention_mask=full_inputs.get("attention_mask"),
        pixel_values=full_inputs.get("pixel_values"),
        image_grid_thw=full_inputs.get("image_grid_thw"),
    )
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    mask = shift_labels.ne(-100)
    target_logprobs = logprobs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    summed = (target_logprobs * mask).sum().item()
    num_tokens = mask.sum().item()
    return float(summed / num_tokens)


def summarize_option_scores(
    option_scores,
    options,
    truth_label,
):
    option_texts = [str(o).strip() for o in options if str(o).strip()]
    if not option_texts:
        raise ValueError("No options provided.")

    scores = {}
    for opt in option_texts:
        if opt not in option_scores:
            raise ValueError(f"Missing score for option: {opt}")
        scores[opt] = float(option_scores[opt])

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    pred, pred_score = ranked[0]
    if len(ranked) >= 2:
        runner_up, runner_up_score = ranked[1]
    else:
        runner_up, runner_up_score = pred, pred_score
    pred_gap = float(pred_score - runner_up_score)

    truth_opt = resolve_label(truth_label, option_texts)
    if not truth_opt:
        raise ValueError("Truth label not in options.")

    truth_score = float(scores[truth_opt])
    other_scores = [v for k, v in scores.items() if k != truth_opt]
    best_other_score = float(max(other_scores)) if other_scores else truth_score
    return {
        "scores": scores,
        "options": option_texts,
        "pred": pred,
        "pred_score": float(pred_score),
        "runner_up": runner_up,
        "runner_up_score": float(runner_up_score),
        "pred_gap": pred_gap,
        "gap": pred_gap,
        "truth_option": truth_opt,
        "truth_score": truth_score,
        "truth_gap": float(truth_score - best_other_score),
        "best_other_score": best_other_score,
        "margin": float(best_other_score - truth_score),
    }


def format_option_scores(score_map):
    if not score_map:
        return ""
    ordered = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    return " ".join(f"{k}={v:.3f}" for k, v in ordered)


def attach_reward_fields(scores_summary):
    scores_summary = dict(scores_summary)
    scores_summary["reward_source"] = "margin"
    scores_summary["prediction_source"] = "pred"
    scores_summary["chosen_pred"] = str(scores_summary.get("pred") or "")
    scores_summary["reward"] = float(scores_summary.get("margin", 0.0))
    return scores_summary


def compute_scores(
    vlm,
    vlm_proc,
    image,
    system_prompt,
    question_text,
    options,
    truth_label,
    *,
    reasoning_off: bool = False,
):
    option_texts = [str(o).strip() for o in options if str(o).strip()]
    raw_scores = {}
    for opt in option_texts:
        raw_scores[opt] = score_candidate(
            vlm,
            vlm_proc,
            image,
            system_prompt,
            question_text,
            opt,
            reasoning_off=reasoning_off,
        )

    if not raw_scores:
        raise ValueError("No options to score.")

    summary = summarize_option_scores(
        raw_scores,
        option_texts,
        truth_label,
    )
    return attach_reward_fields(summary)
