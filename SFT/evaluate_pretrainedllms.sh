# Evaluate the SFT trained mistral model (trained with 0 turn inserted between the query and preference), evaluate on 5 turns inserted
python vllm_infer.py --ft=across_topics_train_with_0turn_mistral --turn=5
python extract_preference_following_accuracy.py --turn=5 --ft=across_topics_train_with_0turn_mistral

# Evaluate the SFT trained mistral model (trained with 5 turn inserted between the query and preference), evaluate on 5 turns inserted
python vllm_infer.py --ft=across_topics_train_with_5turn_mistral --turn=5 
python extract_preference_following_accuracy.py --turn=5 --ft=across_topics_train_with_5turn_mistral

# Evaluate the SFT trained mistral model (trained with 10 turn inserted between the query and preference), evaluate on 5 turns inserted
python vllm_infer.py --ft=across_topics_train_with_10turn_mistral --turn=5 
python extract_preference_following_accuracy.py --turn=5 --ft=across_topics_train_with_10turn_mistral