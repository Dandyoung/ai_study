import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import matplotlib as mpl
import matplotlib.font_manager as fm

# 데이터 로드
def load_data():
  with open('test.txt', 'r', encoding='utf-8') as file:
      data = file.readlines()
  return data

# 전처리: 개행 문자 제거  
def preprocess_data(data):
  return [line.strip() for line in data]

# 불용어 제거
def remove_stopwords(data):
  stopwords = ['은', '는', '이', '가', '을', '를', '에서', '다', '.', '들은']
  filtered_data = []
  for sent in data:
      temp = []
      for word in sent.split():
          if word not in stopwords:
              temp.append(word) 
      filtered_data.append(temp)
  return filtered_data

# 바이그램 생성
def create_bigrams(filtered_data):
  bigrams = []
  for words_list in filtered_data:
      for i in range(len(words_list) - 1):
          for j in range(i+1, len(words_list)):
              bigrams.append([words_list[i], words_list[j]])
              bigrams.append([words_list[j], words_list[i]])
  return bigrams

# 고유 단어 목록 생성
def get_unique_words(bigrams):
  all_words = []
  for bi in bigrams:
      all_words.extend(bi)
  all_words = list(set(all_words))
  all_words.sort()
  return all_words

# 단어 사전 생성
def create_word_dict(all_words):
  return {word: idx for idx, word in enumerate(all_words)}

# 원-핫 인코딩
def create_onehot_encoding(all_words, words_dict):
  onehot_data = np.zeros((len(all_words), len(all_words)))
  for i in range(len(all_words)):
      onehot_data[i][i] = 1
  
  onehot_dict = {}
  for word in all_words:
      onehot_dict[word] = onehot_data[words_dict[word]]
  return onehot_dict

# X, Y 데이터 생성
def create_xy_data(bigrams, onehot_dict):
  X = []
  Y = []
  for bi in bigrams:
      X.append(onehot_dict[bi[0]])
      Y.append(onehot_dict[bi[1]])
  return np.array(X), np.array(Y)

# 모델 생성
def create_model(input_dim, embed_size):
  model = Sequential([
      Dense(embed_size, activation='linear'),
      Dense(input_dim, activation='softmax')
  ])
  model.compile(loss='categorical_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy'])
  return model

def main():
   # 나눔 폰트 설치
   os.system('apt-get update -y')
   os.system('apt-get install -y fonts-nanum')
   
   # 한글 폰트 설정
   font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
   font_prop = fm.FontProperties(fname=font_path)
   
   print("데이터 로딩 중...")
   data = load_data()
   
   print("데이터 전처리 중...")
   processed_data = preprocess_data(data)
   filtered_data = remove_stopwords(processed_data)
   
   print("바이그램 생성 중...")
   bigrams = create_bigrams(filtered_data)
   all_words = get_unique_words(bigrams)
   
   print(f"고유 단어 수: {len(all_words)}")
   words_dict = create_word_dict(all_words)
   onehot_dict = create_onehot_encoding(all_words, words_dict)
   
   print("학습 데이터 준비 중...")
   X, Y = create_xy_data(bigrams, onehot_dict)
   
   print("모델 학습 및 이미지 생성 중...")
   embed_size = 2
   model = create_model(Y.shape[1], embed_size)

   # plots 디렉토리 생성
   if os.path.exists('plots'):
       import shutil
       shutil.rmtree('plots')
   os.makedirs('plots')
   
   # 학습하면서 매 에포크마다 이미지 저장
   epochs = 100
   for epoch in tqdm(range(epochs)):
       model.fit(X, Y, epochs=1, batch_size=256, verbose=1)
       
       # 매 에포크마다 이미지 저장
       weights = model.get_weights()[0]
       plt.figure(figsize=(10, 10))
       
       for word in words_dict:
           idx = words_dict[word]
           x, y = weights[idx]
           plt.scatter(x, y)
           plt.annotate(word, (x, y), fontproperties=font_prop, fontsize=12)
       
       plt.title(f'Epoch {epoch}', fontproperties=font_prop, fontsize=15)
       plt.savefig(f'plots/img_{epoch}.png', dpi=100, bbox_inches='tight')
       plt.close()
   
   print("동영상 생성 중...")
   image_paths = [f'plots/img_{i}.png' for i in range(epochs)]
   
   first_image = cv2.imread(image_paths[0])
   height, width, _ = first_image.shape
   
   fourcc = cv2.VideoWriter_fourcc(*'MP4V')
   video_writer = cv2.VideoWriter('word_embeddings.mp4', fourcc, 30, (width, height))
   
   for image_path in tqdm(image_paths, desc="동영상 생성"):
       image = cv2.imread(image_path)
       if image is not None:
           video_writer.write(image)
       else:
           print(f"Warning: Could not read image {image_path}")
   
   video_writer.release()
   print("완료! 'word_embeddings.mp4' 파일이 생성되었습니다.")

if __name__ == "__main__":
   main()