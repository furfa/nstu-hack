from tqdm import tqdm
import pickle
import face_recognition

class BD:
    def __init__(self):
        self.names = list()
        self.paths = list()
        self.encoded = None
        
    def add(self, name, image_path):
        self.names.append(name)
        self.paths.append(image_path)
    
    def get_names(self):
        return self.names
    
    def get_files(self):
        return self.paths
    
    def make_encodings(self):
        self.encoded = list()
        for n,p in tqdm( zip(self.names, self.paths), total=len(self.paths) ):
            
            
            try:            
                img = face_recognition.load_image_file(p)
                
                for i in range(4):
                    img = rotate(img, -90) 
                    enc_list = face_recognition.face_encodings(img)
                    
                    if len(enc_list) > 0:
                        break
                        
                enc = enc_list[0]
                    
            except IndexError:
                print("Error in " + n + " path: " + p + "(Нет лица)")
                enc = None
                
            self.encoded.append(enc)
            
    def get_data(self):
        return self.names, self.encoded