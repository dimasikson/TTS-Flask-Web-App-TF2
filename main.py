
from flask import Flask, render_template, url_for, request, redirect, session
from datetime import datetime
import string
import time
import datetime
from load import init, synthesize

app = Flask(__name__)

print(datetime.datetime.now(), " model loading")
# ttm_model, ssrn_model = init()
print(datetime.datetime.now(), " model loaded")

@app.route('/', methods=['POST','GET'])
def index():
    text_input = ''
    audio_filename = ''
    text_input_list = []

    return render_template('index.html', text_input=text_input, text_input_list=text_input_list,
                            audio_filename=audio_filename)

 

@app.route('/submit', methods=['POST','GET'])
def submit():
    print('new submit request')
    text_input = ''
    text_input_list = []

    try:
        session['text_input_prev']
    except KeyError:
        session['text_input_prev'] = None

    if session['text_input_prev'] is None:
        session['text_input_prev'] = ''

    try:
        session['audio_filename']
    except KeyError:
        session['audio_filename'] = None

    if session['audio_filename'] is None:
        session['audio_filename'] = ''

    if request.method == 'POST':
        text_input = request.form['content'].lower()
        exclude = set(string.punctuation)
        text_input = ''.join(ch for ch in text_input if ch not in exclude)
        for ch in text_input:
            text_input_list.append(ch)

        if text_input != '' and text_input != session['text_input_prev']:
            session['text_input_prev'] = text_input
            print('loading')

            # synthesize speech from text
            print(datetime.datetime.now())
            output_filename = synthesize(text_input)
            # output_filename = synthesize(text_input, ttm_model, ssrn_model)
            session['audio_filename'] = output_filename
            # session['audio_filename'] = 'test_wav.wav'
            print(datetime.datetime.now())

    return render_template('index.html', text_input=text_input, text_input_list=text_input_list,
                            audio_filename=session['audio_filename'])
 

if __name__ == "__main__":
    app.secret_key = 'SECRET KEY'
    app.run(debug=True,use_reloader=False)

