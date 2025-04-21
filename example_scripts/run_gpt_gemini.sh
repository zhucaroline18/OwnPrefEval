# gpt o4
python ../generation_task/benchmark_generation.py --model=gpt-o4-mini --task=zero-shot --topic=travel_restaurant --inter_turn=8
python ../generation_task/llm_based_evaluation_errortypes.py --model=gpt-o4-mini --task=zero-shot --topic=travel_restaurant --inter_turn=8 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gpt-o4-mini --task=zero-shot --topic=travel_restaurant --inter_turn=8 --pref_form=explicit


python ../generation_task/benchmark_generation.py --model=gpt-o4-mini --task=zero-shot --topic=travel_restaurant --inter_turn=298
python ../generation_task/llm_based_evaluation_errortypes.py --model=gpt-o4-mini --task=zero-shot --topic=travel_restaurant --inter_turn=298 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gpt-o4-mini --task=zero-shot --topic=travel_restaurant --inter_turn=298 --pref_form=explicit



python ../generation_task/benchmark_generation.py --model=gpt-o4-mini --task=remind --topic=travel_restaurant --inter_turn=8
python ../generation_task/llm_based_evaluation_errortypes.py --model=gpt-o4-mini --task=remind --topic=travel_restaurant --inter_turn=8 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gpt-o4-mini --task=remind --topic=travel_restaurant --inter_turn=8 --pref_form=explicit


python ../generation_task/benchmark_generation.py --model=gpt-o4-mini --task=remind --topic=travel_restaurant --inter_turn=298
python ../generation_task/llm_based_evaluation_errortypes.py --model=gpt-o4-mini --task=remind --topic=travel_restaurant --inter_turn=298 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gpt-o4-mini --task=remind --topic=travel_restaurant --inter_turn=298 --pref_form=explicit

# gemini 2.5

python ../generation_task/benchmark_generation.py --model=gemini2.5-flash --task=zero-shot --topic=travel_restaurant --inter_turn=8
python ../generation_task/llm_based_evaluation_errortypes.py --model=gemini2.5-flash --task=zero-shot --topic=travel_restaurant --inter_turn=8 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gemini2.5-flash --task=zero-shot --topic=travel_restaurant --inter_turn=8 --pref_form=explicit


python ../generation_task/benchmark_generation.py --model=gemini2.5-flash --task=zero-shot --topic=travel_restaurant --inter_turn=298
python ../generation_task/llm_based_evaluation_errortypes.py --model=gemini2.5-flash --task=zero-shot --topic=travel_restaurant --inter_turn=298 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gemini2.5-flash --task=zero-shot --topic=travel_restaurant --inter_turn=298 --pref_form=explicit



python ../generation_task/benchmark_generation.py --model=gemini2.5-flash --task=remind --topic=travel_restaurant --inter_turn=8
python ../generation_task/llm_based_evaluation_errortypes.py --model=gemini2.5-flash --task=remind --topic=travel_restaurant --inter_turn=8 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gemini2.5-flash --task=remind --topic=travel_restaurant --inter_turn=8 --pref_form=explicit


python ../generation_task/benchmark_generation.py --model=gemini2.5-flash --task=remind --topic=travel_restaurant --inter_turn=298
python ../generation_task/llm_based_evaluation_errortypes.py --model=gemini2.5-flash --task=remind --topic=travel_restaurant --inter_turn=298 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gemini2.5-flash --task=remind --topic=travel_restaurant --inter_turn=298 --pref_form=explicit



# print all the results
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gpt-o4-mini --task=zero-shot --topic=travel_restaurant --inter_turn=8 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gpt-o4-mini --task=zero-shot --topic=travel_restaurant --inter_turn=298 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gpt-o4-mini --task=remind --topic=travel_restaurant --inter_turn=8 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gpt-o4-mini --task=remind --topic=travel_restaurant --inter_turn=298 --pref_form=explicit


python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gemini2.5-flash --task=zero-shot --topic=travel_restaurant --inter_turn=8 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gemini2.5-flash --task=zero-shot --topic=travel_restaurant --inter_turn=298 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gemini2.5-flash --task=remind --topic=travel_restaurant --inter_turn=8 --pref_form=explicit
python ../generation_task/get_preference_following_accuracy_generation_task.py --model=gemini2.5-flash --task=remind --topic=travel_restaurant --inter_turn=298 --pref_form=explicit
