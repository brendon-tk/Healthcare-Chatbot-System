from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
#from data import Articles

app = Flask(__name__)
# Or app.debug = True or to use it into main
app = Flask(__name__)


#Articles = Articles()


@app.route("/")
def home(): 
	return render_template("home.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/articles')
def articles():
    return render_template('article.html, articles = Articles')





if __name__ == '__main__':
    app.run(debug =True)

