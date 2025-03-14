# app.py
from flask import Flask, render_template, request, jsonify, session, send_from_directory
import random
import os
import numpy as np
import pickle
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# BASE_PATH = '/home/ubuntu/video/AnimateDiff/__assets__/filtered_webvid_datasets_subset_template_text_50'
# BASE_PATH = '/home/ubuntu/video/AnimateDiff/__assets__/filtered_webvid_datasets_subset_template_symbols_50'
BASE_PATH = '/home/ubuntu/video/AnimateDiff/__assets__/filtered_webvid_datasets_subset_template_photorealistic_50'
CLUSTERS = [d for d in os.listdir(BASE_PATH) if d.startswith('filtered_webvid_dataset_')]

def get_random_videos(cluster, num_videos=9):
    """Get random videos from a cluster"""
    video_files = [f for f in os.listdir(os.path.join(BASE_PATH, cluster)) if f.endswith('.gif')]
    selected = random.sample(video_files, num_videos)
    print(f"Selected videos from {cluster}:", selected)  # Debug print
    return selected

def get_cluster_metric(cluster):
    """Get the mean variance embedding from metadata.pkl"""
    metadata_path = os.path.join(BASE_PATH, cluster, 'metadata.pkl')
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        return float(1e5*np.mean(metadata["variance_embedding"]))
    except Exception as e:
        print(f"Error loading metadata for {cluster}: {e}")
        return 0.0

def get_random_clusters(num_clusters=3):
    """Get random clusters"""
    return random.sample(CLUSTERS, num_clusters)

@app.route('/')
def index():
    if 'study_id' not in session:
        session['study_id'] = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{random.randint(1000,9999)}'
    return render_template('index.html')

@app.route('/video/<cluster>/<filename>')
def serve_video(cluster, filename):
    """Serve video files"""
    try:
        print(f"Serving video: {cluster}/{filename}")  # Debug print
        return send_from_directory(os.path.join(BASE_PATH, cluster), filename)
    except Exception as e:
        print(f"Error serving video {cluster}/{filename}: {e}")  # Debug print
        return "File not found", 404

@app.route('/get_videos')
def get_videos():
    clusters = get_random_clusters(3)
    videos_by_cluster = {}
    metrics_by_cluster = {}
    
    for cluster in clusters:
        videos = get_random_videos(cluster)
        videos_by_cluster[cluster] = videos
        metrics_by_cluster[cluster] = get_cluster_metric(cluster)
        
    print("Response data:", {  # Debug print
        'videos': videos_by_cluster,
        'metrics': metrics_by_cluster
    })
    
    return jsonify({
        'videos': videos_by_cluster,
        'metrics': metrics_by_cluster
    })

@app.route('/submit_ranking', methods=['POST'])
def submit_ranking():
    data = request.json
    data['study_id'] = session['study_id']
    data['timestamp'] = datetime.now().isoformat()
    print("Received ranking:", data)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)