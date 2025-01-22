# Train mistral models on it's own completion

# train with 10 turns inserted in the training prompts between the query and the preference.
python train_sft.py --inter_turn=10

# Test with 10 turns inserted between the query and preference.
python vllm_infer.py --ft=across_topics_train_with_10turn --turn=10
python extract_preference_following_accuracy.py --turn=10 --ft=across_topics_train_with_10turn