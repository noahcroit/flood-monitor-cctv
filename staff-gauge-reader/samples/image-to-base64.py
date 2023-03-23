import base64

img_path="sample-staffgauge.JPG"
with open(img_path, "rb") as image2string:
    converted_string = base64.b64encode(image2string.read())
print(converted_string)
