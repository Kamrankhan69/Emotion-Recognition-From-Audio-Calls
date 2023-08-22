from flask import Flask, request, jsonify, render_template, redirect, Response
import os
import librosa
import numpy as np
from scipy.io import wavfile
from playsound import playsound
from tensorflow.keras.models import  model_from_json
import resampy
import statistics
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from collections import Counter
#import whisper
import openai

API_KEY = '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'
model_id = 'whisper-1'

openai.api_key = "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"


#trans_model = whisper.load_model("base")



#import noisereduce as nr
#from pydub import AudioSegment as am

UPLOAD_FOLDER = './static/'
app = Flask(__name__) #Initialize the flask App
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

f = []
json_file = open('./static/cnnModel8020.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("./static/cnnModel8020.h5")
print("Loaded model from disk")


def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate= librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    if chroma:
      stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
      mfccs=np.mean(librosa.feature.mfcc (y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
      result=np.hstack((result, mfccs))
    if chroma:
      chroma=np.mean(librosa.feature.chroma_stft (S=stft, sr=sample_rate).T,axis=0)
      result=np.hstack((result, chroma))
    if mel:
      mel=np.mean(librosa.feature.melspectrogram (y=X, sr=sample_rate). T, axis=0)
      result=np.hstack((result, mel))
    return result

def extract_feature2(audio, sr,  mfcc=True, chroma=True, mel=True):
    if chroma:
        stft = np.abs(librosa.stft(audio))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        result = np.hstack((result, mel))
    return result


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        global file
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            #filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        
        global audFile
        audFile = f'./static/{file.filename}'
        print(audFile)
        f.append(f'./static/{file.filename}')
        print(f)
        #emotions={'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
        em={0 :'anger', 1:'boredom',2:'disgust',3:'fearful',4:'happiness',5:'neutral',6:'sadness',7:'surprise'}
        x =[]

        # Set the duration of the audio chunks (in seconds)
        chunk_duration = 3
        
        audio, sr = librosa.load(audFile)

        # Check if the audio length is greater than 3 seconds
        if len(audio) > sr * 3:
            # Initialize the dictionary to store the features and labels
            features_dict = {}
            # Break the audio into 3-second chunks
            chunks = librosa.util.frame(audio, frame_length=chunk_duration*sr, hop_length=chunk_duration*sr)
            # Extract features from each chunk and add the label and feature to the dictionary
            for i, chunk in enumerate(chunks.T):
                feature = extract_feature2(chunk, sr)
                label = f'{audFile.split("/")[-1]}_{i+1}'
                features_dict[label] = feature.tolist()

            predictions = []

            # Loop over the dictionary keys
            for key in features_dict:
                # Get the values from the key
                values = features_dict[key]
                values = np.array(values).reshape(1, 180, 1)
                # Pass the values to the predict function
                prediction = np.argmax(model.predict(values))

                # Append the prediction to the list of predictions
                predictions.append(prediction)

            mapped_arr = [em[x] for x in predictions]
            most_frequent = statistics.mode(mapped_arr)

            print(most_frequent)

            # pie chart code 
            # Count the occurrences of each emotion
            emotion_counts = Counter(mapped_arr)

            # Get the labels and values for the pie chart
            labels = list(emotion_counts.keys())
            values = list(emotion_counts.values())

            # Compute the percentages
            total = sum(values)
            percentages = [(value / total) * 100 for value in values]

            # Create the pie chart
            fig, ax = plt.subplots(figsize=(8, 7))
            wedges, texts, autotexts = ax.pie(values,
                labels=labels,
                autopct='%1.1f%%',
                startangle=1,
                counterclock=False, # Set the chart to be clockwise
                wedgeprops=dict(width=0.6, edgecolor='w')) # Add white borders to the wedges

            # Modify the legend labels to include the percentage values
            legend_labels = [f'{label} ({percent:.1f}%)' for label, percent in zip(labels, percentages)]


            # Add a title
            ax.set_title('Distribution of Emotions')

            # Add a legend
            ax.legend(wedges, legend_labels,
                    title='Emotions',
                    loc='center',
                    bbox_to_anchor=(0, 0.1),
                    fancybox=True,
                    shadow=True)
            
            
            # Convert the figure to PNG format
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            """
            # HERE THE CODE IS WRITTEN FOR THE TRANSCRIPTION PART
            media_file = open(audFile, 'rb')

            response = openai.Audio.translate(
                api_key=API_KEY,
                model=model_id,
                file=media_file,
                prompt=''
            )


            # HERE IS THE CODE WRITTEN FOR ASPECT PART

            transcribed_text = response['text']
    
            prompt_aspect = f"{transcribed_text} Provide me three important aspects, that are critical for business analysis, provide me the exact aspect(do not add anything from the provided data such as addresses, numbers, names, or locations) rather than general one. List the aspects and do not define them"
    
            result_score_turbo = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.5, 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that finds the aspects"},
                    {"role": "user", "content": prompt_aspect},
                ]
            )
    
            score_turbo = result_score_turbo.choices[0].message.content.strip()   
            """
            return Response(output.getvalue() ,  mimetype='image/png')
          
            
        
            


        else:
            print("Audio length is less than 3 seconds.")



            feature=extract_feature(audFile, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            x = np.array(x)
            predictions = np.argmax(model.predict(x),axis=-1)
            print(predictions)
            n=predictions[0]
            print(n)
            res = em[n]
            return render_template('index.html', prediction_text=str(res))


@app.route('/play',methods=['POST'])
def play():
    print(f)
    ROOT_DIR = os.path.abspath(f[-1])
    root = str(ROOT_DIR)
    print(root)
    root = root.replace("\\","\\\\")
    print(root)
    aud= root
    if request.method == 'POST':
        try:
            audio = playsound(aud , True)
        except:
            audio = playsound(aud , True)
    return render_template('index.html', sound=audio)
    
@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
            if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            #filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            audFile2 = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            
        """
    
    media_file = open(audFile, 'rb')

    response = openai.Audio.translate(
        api_key=API_KEY,
        model=model_id,
        file=media_file,
        prompt=''
        )

        #result = trans_model.transcribe(audFile2 , language = "ur" , task = "translate")
        #transcription = result['text']
    
    instructor = f"""Aspect Analyzer, perform aspect analysis, Provide me three important aspects, that are critical for business analysis, provide me the exact aspect(do not add anything from the provided data such as addresses, numbers, names, or locations) rather than general one. List the aspects and do not define them.""" 

    prompt_aspect = f"""Act as {instructor} 
                    Text for aspect analysis: 
                    {response["text"]}"""
    
    result_score_turbo = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5, 
        messages=[
            {"role": "system", "content": "You are a helpful assistant that finds the aspects"},
            {"role": "user", "content": prompt_aspect},
            ]
        )
    
    score_turbo = result_score_turbo.choices[0].message.content.strip()       

    return render_template('index.html', prediction_text = response['text'], score_turbo=score_turbo)


# @app.route('/analyze',methods=['POST'])
# def analyze():
#     transcribed_text = request.form['transcription']
#     #prompt_aspect = f"Aspects of the {transcribed_text} and only give the words or phrases that describe the aspects of the given text. Do not describe any aspect and just make a list of aspects"
#     prompt_aspect = f"{transcribed_text} Provide me three important aspects, that are critical for business analysis, provide me the exact aspect(do not add anything from the provided data such as addresses, numbers, names, or locations) rather than general one. list aspects in a single sentence"
    
#     result_score_turbo = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         temperature=0.5, 
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that finds the aspects"},
#             {"role": "user", "content": prompt_aspect},
#             ]
#         )
    
#     score_turbo = result_score_turbo.choices[0].message.content.strip()    
#     return render_template('index.html',  score_turbo=score_turbo)


@app.route('/nextpage')
def nextpage():
    return render_template('main.html')
 

if __name__ == "__main__":
    app.debug = True
    app.run()
