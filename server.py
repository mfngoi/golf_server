from flask import Flask, request, jsonify, abort
from Golf_Analyser import video_analyzer
import os

app = Flask(__name__)

@app.route('/')
def homepage():
    print("This is your homepage")
    return "This is your homepage"

@app.route('/hello')
def hello():
    print("Greetings!")
    return "Greetings!"

@app.route('/analyser', methods=['GET', 'POST'])
def golf_analyser():

    print("request recieved")

    try:

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
        print(links)

        # Once the videos have been analyzed, delete them
        os.remove(filename1)
        os.remove(filename2)
    
    except Exception as e:
        print(e)

    return jsonify(links)


if __name__ == "__main__":
    app.run()