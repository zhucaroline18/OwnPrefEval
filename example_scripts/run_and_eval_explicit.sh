# Generation Tasks example:

# Benchmark Claude 3 Haiku with zero-shot on the explicit preferences and with 2 inter turns inserted
python ../generation_task/benchmark_generation.py --model=claude3hk --task=zero-shot --topic=travel_restaurant --inter_turn=2

# LLM-based evaluation of Claude 3 Haiku with zero-shot on the explicit preferences and with 2 inter turns inserted
python ../generation_task/llm_based_evaluation_errortypes.py --model=claude3hk --task=zero-shot --topic=travel_restaurant --inter_turn=2 --pref_form=explicit

# Print evaluation results:
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=claude3hk --task=zero-shot --topic=travel_restaurant --inter_turn=2 --pref_form=explicit

