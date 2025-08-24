import cv2
import os
import shutil
from flask import Flask, request, render_template, redirect
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import json

app = Flask(__name__)
nimgs = 30
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray, threshold=0.75):  # Increased threshold!
    modelpath = 'static/face_recognition_model.pkl'
    if not os.path.exists(modelpath):
        return "Not in Database"
    model = joblib.load(modelpath)
    probs = model.predict_proba(facearray)[0]
    best_idx = np.argmax(probs)
    best_label = model.classes_[best_idx]
    best_conf = probs[best_idx]
    if best_conf >= threshold:
        return best_label
    else:
        return "Not in Database"

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    if len(faces) == 0:
        return
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)
    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, l

def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser + '/' + i)
    os.rmdir(duser)

################## ROUTING FUNCTIONS #########################
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    userlist, _, _, _ = getallusers()
    mess = request.args.get('mess')
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
        totalreg=totalreg(), datetoday2=datetoday2, mess=mess,
        users_json=json.dumps(userlist))

@app.route('/showusers')
def showusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', names=names, rolls=rolls, l=l, datetoday2=datetoday2)

@app.route('/delete_selected_users', methods=['POST'])
def delete_selected_users():
    users_to_delete = request.form.getlist('users_to_delete')
    for u in users_to_delete:
        folder = os.path.join('static/faces', u)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    if os.path.exists('static/face_recognition_model.pkl'):
        train_model()
    return redirect('/')

@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)
    if os.listdir('static/faces/') == []:
        if os.path.exists('static/face_recognition_model.pkl'):
            os.remove('static/face_recognition_model.pkl')
    try:
        train_model()
    except:
        pass
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l,
        totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/delete_all_users', methods=['POST'])
def delete_all_users():
    faces_dir = 'static/faces'
    if os.path.isdir(faces_dir):
        for folder in os.listdir(faces_dir):
            shutil.rmtree(os.path.join(faces_dir, folder))
    model_path = 'static/face_recognition_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
    names, rolls, times, l = extract_attendance()
    mess = "All user data deleted successfully."
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
        totalreg=totalreg(), datetoday2=datetoday2, mess=mess,
        users_json=json.dumps([]))

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
            totalreg=totalreg(), datetoday2=datetoday2,
            mess='There is no trained model in the static folder. Please add a new face to continue.',
            users_json=json.dumps([]))
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            cv2.putText(frame, identified_person, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if identified_person != "Not in Database":
                add_attendance(identified_person)
        cv2.imshow('Attendance', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    userlist, _, _, _ = getallusers()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
        totalreg=totalreg(), datetoday2=datetoday2, users_json=json.dumps(userlist))

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
            if j == nimgs * 5 or i >= nimgs:
                break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if i >= nimgs:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    userlist, _, _, _ = getallusers()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
        totalreg=totalreg(), datetoday2=datetoday2, users_json=json.dumps(userlist))

if __name__ == '__main__':
    app.run(debug=True)
