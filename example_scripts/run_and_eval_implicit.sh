#"""Persona-Driven Implicit Preferences"""

# Benchmark Claude 3 Haiku in zero-shot mode with persona-driven implicit preferences and 2 inter-turns
python ../generation_task/benchmark_generation.py --model=claude3hk --task=zero-shot --topic=travel_restaurant --inter_turn=2 --pref_form=implicit --pref_type=persona

# Evaluate Claude 3 Haiku in zero-shot mode for persona-driven implicit preferences with 2 inter-turns
python ../generation_task/llm_based_evaluation_errortypes.py --model=claude3hk --task=zero-shot --topic=travel_restaurant --inter_turn=2 --pref_form=implicit --pref_type=persona

# Display evaluation results for preference adherence in the generation task
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=claude3hk --task=zero-shot --topic=travel_restaurant --inter_turn=2 --pref_form=implicit --pref_type=persona

#"""Choice-Based Implicit Preferences"""

# Benchmark Claude 3 Haiku in zero-shot mode with choice-based implicit preferences and 2 inter-turns
python ../generation_task/benchmark_generation.py --model=claude3hk --task=zero-shot --topic=travel_restaurant --inter_turn=2 --pref_form=implicit --pref_type=choice

# Evaluate Claude 3 Haiku in zero-shot mode for choice-based implicit preferences with 2 inter-turns
python ../generation_task/llm_based_evaluation_errortypes.py --model=claude3hk --task=zero-shot --topic=travel_restaurant --inter_turn=2 --pref_form=implicit --pref_type=choice

# Display evaluation results for preference adherence in the generation task
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=claude3hk --task=zero-shot --topic=travel_restaurant --inter_turn=2 --pref_form=implicit --pref_type=choice

wait
