# Day 1


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


# Day 2


1)Доделать трекинг лиц и подркутить туда face rec

2) Засплитить выборку на учителей и студентов

3) Трекать и добавлять в БД данные: Время + статус+имя
    P.S нужно добавлять только во время первого детектинга и дальше трекать,
    пока не потеряем. Если не зайдет трекинг, можно просто раз в n-время добавлять, если человек уже появился 
