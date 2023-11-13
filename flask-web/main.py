from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
#from werkzeug.datastructures import  FileStorage
import errno
import os
from datetime import datetime
from pyresparser import ResumeParser

app = Flask(__name__, template_folder='template')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file11():
   if request.method == 'POST':
      f = request.files['file']
      mydir = os.path.join(
        os.getcwd(), 'resume', 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
      #fnm = os.path.join(mydir, f.filename)
      #path_items = fnm.split(os.sep)
      #fnm_path = '/'.join(path_items)
      #os.makedirs(mydir, exist_ok=True)
      os.makedirs(mydir, mode=777, exist_ok=True)
      fpp = os.path.join(mydir, secure_filename(f.filename))
      f.save(fpp)
      data = ResumeParser(fpp).get_extracted_data()
      return data
   else:
      return render_template('upload.html')

def filecreation(list, filename):
   mydir = os.path.join(
      os.getcwd(), 
      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
   try:
      os.makedirs(mydir)
   except OSError as e:
      if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..
   with open(os.path.join(mydir, filename), 'w') as d:
      d.writelines(list)