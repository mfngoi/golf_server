from flask import Flask, request, jsonify, abort
from Golf_Analyser import video_analyzer
import os

app = Flask(__name__)


@app.route('/hello')
def hello():
    return "Greetings!"

@app.route('/')
def homepage():
    return "This is your homepage"

@app.route('/analyser', methods=['GET', 'POST'])
def golf_analyser():

    # retrieve videos from the request using "request" in Flask
    video1 = request.files.getlist("video1")[0]
    video2 = request.files.getlist("video2")[0]

    filename1 = video1.filename
    filename2 = video2.filename

    # Check if videos from request are valid
    if filename1 == "" or filename2 == "":
        abort(400)

    # Save the videos from request
    video1.save(filename1)
    video2.save(filename2)

    # Run the video analyzer method
    links = video_analyzer(filename1, filename2)

    # Once the videos have been analyzed, delete them
    os.remove(filename1)
    os.remove(filename2)

    return jsonify(links)


if __name__ == "__main__":
    app.run()