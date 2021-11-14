import os
from flask import Flask, request, json, jsonify
import random
from sentiment_analysis_service import SentimentAnalysisService

app = Flask(__name__)


@app.route('/sentiment/audio',methods = ['POST'])
def audio_analysis():

    request_data = json.loads(request.data)

    print(request_data)

    #base64_wavefile = request_data['audioString']

    ## TODO
    #Save wave file content to some file add add the file to the filelist below

    emotion_index = random.randint(1, 8)

    fileList = []

    # Path of wavefile to be tested , one example here
    #fileName = '/Users/adityaasthana/cs196/test_data/03-01-05-01-01-01-03.wav'
    fileName = '/Users/adityaasthana/cs196/Audio_Speech_Actors_01-24/Actor_01/03-01-08-02-02-01-01.wav'
    for root, dirs, files in os.walk("/Users/adityaasthana/cs196/test_data"):
        for file in files:
            if file.endswith(".wav"):
                fileList.append(os.path.join(root, file))
                print(len(fileList))
    print(fileList)



    emotion_detection_service = SentimentAnalysisService()


    X = emotion_detection_service.predict_result(fileList)


    emotion_index = X.astype(int)

    print('emotion index: ' + str(emotion_index))

    del emotion_detection_service

    #return jsonify(emotionIndex=emotion_index)

    return str(emotion_index)

if __name__ == '__main__':
    print('Running in local environment')
    app.run(host="localhost", port=8000, debug=True)
#    app.run(debug=True, host='ec2-184-72-140-87.compute-1.amazonaws.com')
