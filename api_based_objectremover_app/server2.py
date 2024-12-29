from flask import Flask, request, jsonify , json as JSON
import requests
import base64

app = Flask(__name__)

@app.route('/send_images', methods=['POST'])
def send_images():
    url = "http://127.0.0.1:8888/inpainting"
    image1_path = r"C:\Users\SHRAMADEEP\Desktop\dev\guy.png"
    image2_path = r"C:\Users\SHRAMADEEP\Desktop\dev\masked_image.png"

    # if not url or not image1_path or not image2_path:
    #     return jsonify({"error": "Missing url or image paths"}), 400
    with open(image1_path, 'rb') as img1_file:
        image1 = base64.b64encode(img1_file.read())
    with open(image2_path, 'rb') as img2_file:
        image2 = base64.b64encode(img2_file.read())
        
    headers = {
    "Content-Type": "application/json"
    }   
    payload = {
        "img": image1.decode('utf-8'),
        "masked": image2.decode('utf-8')
    }
    # json_payload = JSON.dumps(payload)
    # json_payload = JSON.dumps(payload)
    # print("it worked1")
    # print(payload)
    response = requests.post(url, json=payload , headers=headers)
    # print("it worked2")
    return jsonify({"status_code": response.status_code, "response": response.text})

if __name__ == '__main__':
    app.run(debug=True)