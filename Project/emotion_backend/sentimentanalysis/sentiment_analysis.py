import os
from flask import Flask, request, json, jsonify, flash
import random
from sentiment_analysis_service import SentimentAnalysisService
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)

@app.route('/sentiment/audio',methods = ['POST'])
def audio_analysis():
    fileList = []
    if 'file' not in request.files:
        print('No file part')
        return str(-1)
    file = request.files['file']
    if file.filename != '':
        file.save(file.filename)
        print(file.filename)
        fileName = '/Users/adityaasthana/cs196/sentimentanalysis/' + file.filename
        print(fileName)
        fileList.append(fileName)
        
        emotion_detection_service = SentimentAnalysisService()

        X = emotion_detection_service.predict_result(fileList)

        emotion_index = X.astype(int)

        print('emotion index: ' + str(emotion_index))

        del emotion_detection_service

        # return jsonify(emotionIndex=emotion_index)

        return str(emotion_index)

if __name__ == '__main__':
    print('Running in local environment')
    app.run(host="localhost", port=8000, debug=True)
#    app.run(debug=True, host='http://909a-2601-243-197e-fa00-f855-46ea-5dfd-4b1b.ngrok.io')