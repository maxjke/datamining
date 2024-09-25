from flask import render_template, request
from FlaskWebProject1 import app
import mysql.connector  
import numpy as np
from scipy.spatial import distance
import json


db_config = {
    'host': 'localhost',
    'user': 'root',  
    'password': 'root',  
    'database': 'knn_classifier'
}


def get_data_from_db():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT x1, x2, label FROM points")
    rows = cursor.fetchall()
    data_points = [(row[0], row[1]) for row in rows]
    labels = [row[2] for row in rows]
    cursor.close()
    conn.close()
    return data_points, labels


def euc_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def manhattan_distance(a, b):
    return distance.cityblock(a, b)


def knn_classify(new_point, data_points, labels, k=3, metric='euclidean'):
    distances = []
    for point, label in zip(data_points, labels):
        if metric == 'euclidean':
            dist = euc_distance(new_point, point)
        elif metric == 'manhattan':
            dist = manhattan_distance(new_point, point)
        distances.append((dist, label))
    
    
    distances = sorted(distances, key=lambda x: x[0])
    nearest_neighbors = [label for _, label in distances[:k]]
    
    
    return max(set(nearest_neighbors), key=nearest_neighbors.count)


@app.route('/')
def index():
    
    data_points, labels = get_data_from_db()
    data_json = json.dumps({'points': data_points, 'labels': labels})
    return render_template('index.html', data_points=data_points, labels=labels, data_json=data_json)


@app.route('/classify', methods=['POST'])
def classify():
    
    data_points, labels = get_data_from_db()

    
    data = request.form.get('data')

    try:
        
        new_point = list(map(float, data.split(',')))
    except ValueError:
        return "Error: Invalid data format", 400

    
    if len(new_point) != 2:
        return "Error: Please enter exactly 2 coordinates.", 400
    
    
    k = 3  
    result = knn_classify(new_point, data_points, labels, k, metric='euclidean')

    
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO points (x1, x2, label) VALUES (%s, %s, %s)", (new_point[0], new_point[1], result))
    conn.commit()
    cursor.close()
    conn.close()

    
    data_points.append(tuple(new_point))
    labels.append(result)
    data_json = json.dumps({'points': data_points, 'labels': labels})

    return render_template('result.html', result=result, new_point=new_point, data_points=data_points, labels=labels, data_json=data_json)
