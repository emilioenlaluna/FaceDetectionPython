from flask import Flask, request, jsonify
import face_recognition
import urllib.request

app = Flask(__name__)

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    # Get image URLs from request
    known_image_url = request.json['known_image_url']
    unknown_image_url = request.json['unknown_image_url']
    
    # Load images from URLs
    known_image = urllib.request.urlopen(known_image_url)
    unknown_image = urllib.request.urlopen(unknown_image_url)
    
    # Convert images to numpy arrays for face_recognition library
    known_image_data = np.asarray(bytearray(known_image.read()), dtype=np.uint8)
    unknown_image_data = np.asarray(bytearray(unknown_image.read()), dtype=np.uint8)
    known_image_np = cv2.imdecode(known_image_data, cv2.IMREAD_COLOR)
    unknown_image_np = cv2.imdecode(unknown_image_data, cv2.IMREAD_COLOR)
    
    # Get face encodings
    biden_encoding = face_recognition.face_encodings(known_image_np)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image_np)[0]
    
    # Compare faces
    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
    
    # Return results
    return jsonify({'result': results[0]})

if __name__ == '__main__':
    app.run(debug=True)
