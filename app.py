from flask import Flask, render_template, url_for, request
from flask_talisman import Talisman

app = Flask(__name__)
csp = {
    'default-src': ['\'self\'', 'data:', 'https://fonts.gstatic.com/', 
    'https://code.getmdl.io', 'https://fonts.googleapis.com', 
    '\'unsafe-inline\'', 'cdn.jsdelivr.net', 'https://gist.github.com/',
    'https://github.githubassets.com/']    
}
Talisman(app, force_https_permanent=True, content_security_policy=csp)

@app.route("/")
def index():
    return render_template("background.html")

@app.route("/lemurnet/")
def lemurnet():
    return render_template("index.html")

@app.route("/links/")
def links():
    return render_template("links.html")

@app.route("/blog_downloading_dataset/")
def blog_downloading_dataset():
    return render_template("blog_downloading_dataset.html")


@app.route("/blog_cleaning_dataset/")
def blog_cleaning_dataset():
    return render_template("blog_cleaning_dataset.html")


@app.route("/blog_training/")
def blog_training():
    return render_template("blog_training.html")

@app.route("/blog_deploying/")
def blog_deploying():
    return render_template("blog_deploying.html")

@app.route("/blog/")
def blog():
    return render_template("blog.html")

if __name__ == "__main__":
    app.run()
