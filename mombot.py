import json, sys
from os.path import join, dirname
from watson_developer_cloud import SpeechToTextV1
import speech_recognition as sr
 
# Record Audio

output_json = 'textfile.json'

speech_to_text = SpeechToTextV1(
    username='58e81a5b-0299-4eb6-992b-e566117950fd',
    password='XP6g7Pr0Ssne',
    x_watson_learning_opt_out=True
)

print(json.dumps(speech_to_text.models(), indent=2))

print(json.dumps(speech_to_text.get_model('en-US_BroadbandModel'), indent=2))
r = sr.Recognizer()
with sr.Microphone() as source:
    audio = r.listen(source)
    result = speech_to_text.recognize(audio, content_type='audio/wav', timestamps=True, word_confidence=True, speaker_labels=True)
    # with open(output_json, 'w') as f:
    #     json.dump(result, f, ensure_ascii=False)
    print(result)