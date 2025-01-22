# MCQ explicit preferences tasks:

all_topics=("travel_transportation" "shop_motors" "lifestyle_beauty" "travel_restaurant" "shop_fashion" 
"entertain_shows" "pet_ownership" "lifestyle_fit" "entertain_games" "shop_home" "lifestyle_health" 
"travel_activities" "education_learning_styles" "entertain_music_book" "professional_work_location_style" 
"education_resources" "lifestyle_dietary" "shop_technology" "travel_hotel" "entertain_sports")

topics=("travel_restaurant")
for topic in "${topics[@]}"; do
    python ../classification_task/benchmark_classification.py --pref_form=explicit --task=zero-shot --model=claude3hk --inter_turns=0 --topic=$topic
done

for topic in "${topics[@]}"; do
    python ../classification_task/benchmark_classification.py --pref_form=implicit --pref_type=choice --task=zero-shot --model=claude3hk --inter_turns=0 --topic=$topic
done

for topic in "${topics[@]}"; do
    python ../classification_task/benchmark_classification.py --pref_form=implicit --pref_type=persona --task=zero-shot --model=claude3hk --inter_turns=0 --topic=$topic
done