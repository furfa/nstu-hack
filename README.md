# nstu-hack


1) Найти людей на видосе
2) face rec на этом же видео
3) path + label
4)
potemin_image = face_recognition.load_image_file("ya.JPG")
potemin_face_encoding = face_recognition.face_encodings(potemin_image)[0]

known_face_encodings = [
    potemin_face_encoding
]
known_face_names = [
    'Potemin'
]
