from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    title = 'title'
    return render_template('index.html',title=title)


