import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request
import json
from flask_cors import CORS, cross_origin
import joblib
import os
from os.path import join, exists, dirname
from flask import render_template
import cv2
import random

# Function to ensure directories exist
def ensure_dir(directory):
    if not exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return directory

train_path = ['home', 'tshirts', 'jeans', 'sweat_shirts', 'shoes', 'googles', 'ties', 'watches', 'shirts']
num_files = []

# Use os.path.join for cross-platform path compatibility
base_path = os.getcwd()
path = join(base_path, 'static')
path = join(path, 'images')
path = join(path, 'test')

# Make sure the base directories exist
ensure_dir(path)

# For images path to be displayed
images = []
for j in range(len(train_path)):
    # Use os.path.join instead of string concatenation for paths
    category_dir = train_path[j]
    dir_path = join(path, category_dir)
    
    # Ensure the directory exists
    ensure_dir(dir_path)
    
    # Build path for web display - use forward slashes for URLs
    sent_path = f"/static/images/test/{category_dir}"
    
    sent_image = []
    try:
        dir_files = os.listdir(dir_path)
        num_files.append(len(dir_files))
        for i in dir_files:
            sent_image.append(join(sent_path, i))
    except FileNotFoundError:
        print(f"Warning: Directory not found: {dir_path}")
        num_files.append(0)
    
    images.append(sent_image)

# If no images were found, create a placeholder
if all(len(img) == 0 for img in images):
    print("Warning: No images found. The application may not work correctly.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET', 'POST'])
def hello():
    return render_template('index.html')

@app.route("/cat", methods=['GET', 'POST'])
def cat():
    directory = int(request.args['id'])
    print('directory ', directory)
    nameofdir = train_path[directory]
    print('nameofdir ', nameofdir)
    img_path = []
    seq = []
    
    # Check if there are images in this directory
    if num_files[directory] > 0:
        for i in range(min(12, num_files[directory])):
            num = random.randrange(0, num_files[directory]) if num_files[directory] > 1 else 0
            seq.append(num)
            img_path.append(images[directory][num])
    else:
        print(f"Warning: No images found in directory {nameofdir}")
    
    return render_template('cat.html', data=img_path, seq=seq, name=nameofdir, did=directory)

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route("/api", methods=['GET', 'POST'])
def make_predict():
    print('start substitute')
    imgno = int(request.args['id'])
    print('imgno is ', imgno)
    dirid = int(request.args['dirid'])
    print('dirid ', dirid)
    nameofdir = train_path[dirid]
    print('nameofdir ', nameofdir)
    li = []
    temp = []
    cluster_assignment = []
    did = []
    
    # Check if files exist before loading
    cluster_assignment_file = f'cluster_assignment{str(dirid)}.pkl'
    cluster_assignmentN_file = 'cluster_assignmentN.pkl'
    
    try:
        # Loading model
        if os.path.exists(cluster_assignment_file):
            cluster_assignment = joblib.load(cluster_assignment_file)
        else:
            print(f"Warning: File not found: {cluster_assignment_file}")
            return render_template('error.html', message=f"Missing model data: {cluster_assignment_file}")
            
        if os.path.exists(cluster_assignmentN_file):
            cluster_assignmentN = joblib.load(cluster_assignmentN_file)
        else:
            print(f"Warning: File not found: {cluster_assignmentN_file}")
            return render_template('error.html', message=f"Missing model data: {cluster_assignmentN_file}")
            
        res = images[dirid][imgno]
        li.append(res)
        temp.append(imgno)
        for i in cluster_assignment[imgno][:6]:
            did.append(dirid)
            li.append(images[dirid][i])
            temp.append(i)

        # Use a cross-platform compatible path for "all" directory
        all_dir = join(path, 'all')
        ensure_dir(all_dir)
        pathN = "/static/images/test/all/"

        for i in cluster_assignmentN[imgno][:6]:
            flag = 0
            for j in range(len(train_path)-1):
                if train_path[j+1] != 'home':
                    img_file_dir = join(path, train_path[j+1])
                    if exists(img_file_dir):
                        img_file = os.listdir(img_file_dir)
                        for img in img_file:
                            if img == str(i)+'.jpg':
                                did.append(j+1)
                                flag = 1
                                break
                if flag == 1:
                    break        
            li.append(join(pathN, str(i)+'.jpg'))
            temp.append(i)        
        print(li)
        return render_template('details.html', list=li, seq=temp, did=did, name=nameofdir)
    except Exception as e:
        print(f"Error in make_predict: {str(e)}")
        return render_template('error.html', message=str(e))

@app.route("/api2", methods=['GET', 'POST'])
def make_complimentary():
    try:
        print('start make complimentary')
        imgno = int(request.args['id'])
        print('imgno is ', imgno)
        dirid = int(request.args['dirid'])
        print('dirid ', dirid)
        nameofdir = train_path[dirid]
        print('nameofdir ', nameofdir)

        li = []
        temp = []
        did = []
        
        # Check if files exist before loading
        cluster_assignment_file = f'cluster_assignment{str(dirid)}.pkl'
        cluster_assignmentN_file = 'cluster_assignmentN.pkl'
        
        if os.path.exists(cluster_assignment_file):
            cluster_assignment = joblib.load(cluster_assignment_file)
        else:
            print(f"Warning: File not found: {cluster_assignment_file}")
            return render_template('error.html', message=f"Missing model data: {cluster_assignment_file}")
            
        if os.path.exists(cluster_assignmentN_file):
            cluster_assignmentN = joblib.load(cluster_assignmentN_file)
        else:
            print(f"Warning: File not found: {cluster_assignmentN_file}")
            return render_template('error.html', message=f"Missing model data: {cluster_assignmentN_file}")
        
        k = 0
        dir_path = join(path, train_path[dirid])
        if exists(dir_path):
            for j, i in enumerate(os.listdir(dir_path)):
                if i == str(imgno) + '.jpg':
                    k = j
                    break
        else:
            print(f"Warning: Directory not found: {dir_path}")
        
        print('cluster is ', cluster_assignment[k])
        res = images[dirid][k]
        li.append(res)
        temp.append(k)
        for i in cluster_assignment[k][:6]:
            did.append(dirid)
            li.append(images[dirid][i])
            temp.append(i)

        # Use a cross-platform compatible path for "all" directory
        pathN = "/static/images/test/all/"

        for i in cluster_assignmentN[imgno][:6]:
            flag = 0
            for j in range(len(train_path)-1):
                if train_path[j+1] != 'home':
                    img_file_dir = join(path, train_path[j+1])
                    if exists(img_file_dir):
                        img_file = os.listdir(img_file_dir)
                        for img in img_file:
                            if img == str(i)+'.jpg':
                                did.append(j+1)
                                flag = 1
                                break
                if flag == 1:
                    break        
            li.append(join(pathN, str(i)+'.jpg'))
            temp.append(i)        
        print(li)
        return render_template('details.html', list=li, seq=temp, did=did, name=nameofdir)
    except Exception as e:
        print(f"Error in make_complimentary: {str(e)}")
        return render_template('error.html', message=str(e))

@app.route("/contactPost", methods=['GET', 'POST'])
def contactPost():
    return render_template('contactPost.html')

if __name__ == '__main__':
    # Create a basic error.html template if it doesn't exist
    templates_dir = join(base_path, 'templates')
    error_template = join(templates_dir, 'error.html')
    if not exists(templates_dir):
        ensure_dir(templates_dir)
    if not exists(error_template):
        with open(error_template, 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
</head>
<body>
    <h1>Error</h1>
    <p>{{ message }}</p>
    <a href="/">Return to Home</a>
</body>
</html>''')
    
    app.run(port=9005, debug=True)
