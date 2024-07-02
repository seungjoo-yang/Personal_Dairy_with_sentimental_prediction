from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup  # dynamically update learning rate
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import random
from html import unescape

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

#회원 DB
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    
#일기목록 DB
class DiaryEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    text = db.Column(db.Text, nullable=False)  # text 컬럼 추가
    analysis_result = db.Column(db.String(50), nullable=True)
    image_path = db.Column(db.String(100), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# BERTSentenceTransform 수정
class BERTSentenceTransform:
    r"""BERT style data transformation.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """

    def __init__(self, tokenizer, max_seq_length,vocab, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = vocab

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
        sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 2 strings:
        text_a, text_b.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens: '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14

        For single sequences, the input is a tuple of single string:
        text_a.

        Inputs:
            text_a: 'the dog is hairy .'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a: '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 2 strings:
            (text_a, text_b). For single sequences, the input is a tuple of single
            string: (text_a,).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)

        """

        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2  # 뒤의 조건이 True가 아니면 AssertError 발생
            text_b = line[1]

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        #vocab = self._tokenizer.vocab
        vocab = self._vocab
        tokens = []
        tokens.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens.append(vocab.sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32')

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        #transform = nlp.data.BERTSentenceTransform(
        #    tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 5,   # 감정 클래스 수
                 dr_rate = None,
                 params = None):

        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p = dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict = False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
def get_kobert_model(model_path, vocab_file, ctx="cpu"):
                bertmodel = BertModel.from_pretrained(model_path)
                device = torch.device(ctx)
                bertmodel.to(device)
                bertmodel.eval()
                vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         padding_token='[PAD]')
                return bertmodel, vocab_b_obj
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

                # Torch GPU 설정
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

bertmodel, vocab = get_kobert_model('skt/kobert-base-v1',tokenizer.vocab_file)
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)
PATH = 'model_state_dict_test3.pt'  # state_dict 저장 경로
model = BERTClassifier(bertmodel,  dr_rate = 0.5)
model.load_state_dict(torch.load(PATH))
model.to(device)

def rader_chart(emotions,date_input, user):
    plt.switch_backend('agg')
    #emotions = [joy, confusion, anger, anxiety, sadness]
    emotion_names = ['기쁨', '당황', '분노', '불안', '슬픔']
    
    for i in range(len(emotions)):
        emotions[i] = int(emotions[i] * 100) + 10  # 인트로 변경 후, 최소값으로 5를 부여
        emotions[i] = min(emotions[i], 100)  # 최대값을 100으로 제한

    today_emotion = {'Emotion':emotion_names,
                    'Value':emotions}
    
    # 데이터프레임으로 변환        
    df_emotion = pd.DataFrame(today_emotion)  
    plt.rc('font', family='Malgun Gothic')
    # 레이더차트 그리기
    label = df_emotion['Emotion']
    value = df_emotion['Value']
    
    num_vars = len(label)
    
    # 각 변수의 각도 계산
    angles = [n / float(num_vars) * 2 * 3.14158 for n in range(num_vars)]
    angles += angles[:1]  # 처음 원소를 다시 추가하여 완전한 원을 그릴 수 있도록 함
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # 차트 생성
    ax.plot(angles, value.tolist() + value.tolist()[:1], color='blue', alpha=0.25)  # value 리스트에 마지막 원소를 처음에 다시 추가함
    ax.fill(angles, value.tolist() + value.tolist()[:1], color='blue', alpha=0.25)  # 내부를 채움
    
    # 외곽 다각형 설정
    ax.set_theta_offset(3.14159 / 2)  # 90도 회전하여 다각형으로 설정
    ax.set_theta_direction(-1)  # 시계 반대 방향으로 다각형 설정
    
    ax.set_yticklabels(['20%', '40%', '60%', '80%','100%'], fontsize=8)  # 반지름 라벨 제거
    plt.xticks(angles[:-1], label, color='black', size=10)  # 각도 위치에 감정 레이블 추가

    # 그래프 크기 400 x 400 픽셀 설정
    x = 400 / fig.dpi  # 가로 길이 
    y = 400 / fig.dpi  # 세로 길이 
    fig.set_figwidth(x)
    fig.set_figheight(y)
    
    plt.rc('font', family='Malgun Gothic')
    plt.title('감정 확률 차트',pad=20)
    image_path = "static/"+date_input+"radar_chart"+user+".png"
    plt.savefig(image_path, format='png')
    plt.close()
    image_data = date_input + 'radar_chart'+user+'.png'
    
    return image_data

# 회원가입 중복체크 하는 함수
def is_username_taken(username):
    return User.query.filter_by(username=username).first() is not None

# 사용자 감정에 따른 음악추천 함수
def music_recommend(emotion):
    # API 키 설정
    api_key = 'AIzaSyCKW6KGpjqk_QExAIr781KeFxXUbjKog6A'

    # api로 클라이언트 생성
    youtube = build('youtube', 'v3', developerKey=api_key)

    # 기분에 따른 검색어 랜덤 추출
    emotion_dict = {'기쁨': '기쁠', '당황': '당황활', '분노': '화가 날', '슬픔': '슬플', '불안': '불안할', '두려움': '두려울'}
    today_emotion = emotion_dict[emotion]
    keyword = f"{today_emotion}때 듣는 노래"
    search_keyword = keyword.replace(" ", "+")

    # youtube에서 검색
    search_response = youtube.search().list(
        q=search_keyword,
        part='id,snippet',
        maxResults=1
    ).execute()

    # 결과 중 첫번째 동영상 정보 가져오기
    first_video = search_response['items'][0]
    music_title =  unescape(first_video['snippet']['title'])
    music_img_url = first_video['snippet']['thumbnails']['default']['url']
    music_id = first_video['id']['videoId']
    music_url = f'https://www.youtube.com/watch?v={music_id}'

    return music_title, music_img_url, music_url

# 감정에 따라 영상 추천하는 함수
def video_recommend(emotion):
    # API 키 설정
    api_key = 'AIzaSyCKW6KGpjqk_QExAIr781KeFxXUbjKog6A'

    # api로 클라이언트 생성
    youtube = build('youtube', 'v3', developerKey=api_key)

    # 기분에 따른 검색어 랜덤 추출
    emotion_dict = {'기쁨': '기쁠', '당황':'당황활', '분노':'화가 날', '슬픔':'슬플', '불안':'불안할', '두려움':'두려울'}
    today_emotion = emotion_dict[emotion]

    positive = ['기쁠']
    negative = ['당황할', '화가 날', '슬플', '불안할', '두려울' ]

    if today_emotion in positive:
        keyword = "즐거운 영상"
    else:
        keyword = f"{today_emotion}때 도움되는 영상"

    search_keyword = keyword.replace(" ","+")

    # youtube에서 검색
    search_response = youtube.search().list(
        q=search_keyword,
        part='id,snippet',
        maxResults=1
    ).execute()

    # 결과 중 첫번째 동영상 정보 가져오기
    first_video = search_response['items'][0]
    video_title =  unescape(first_video['snippet']['title'])
    video_img_url = first_video['snippet']['thumbnails']['default']['url']
    video_id = first_video['id']['videoId']
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    return video_title, video_img_url, video_url

# 감정에 따라 조언이 되는 한마디를 반환하는 함수
def get_one_word(emotion):
    joy = ['당신이 기뻐하는 걸 보니 저도 행복해요!',
           '당신의 긍정적 에너지가 주변을 밝게 만들어요!', 
           '당신이 기뻐하는 것을 보는것만으로도 저도 기분이 좋아요!']
    confusion = ['당황하는건 당연해요. 함께 천천히 숨을 고르며 해결해봐요!',
                '당황스러운 순간에도 우리는 함께 있어요. 조금씩 문제를 해결해 나갈 거예요!',
                '어색한 상황이라 당황할 수 있어요. 조금씩 편안해지도록 노력해보아요!']
    anger = ['이 순간에 대한 분노는 잠시 일시적인 감정일 수 있어요. 조금 시간을 가져도 좋아요!',
            '마음을 가라앉히고 명료한 상황판단을 위해 깊게 숨을 들이마셔보세요!',
            '당신의 감정은 당신의 성장과 이해를 위한 일부일 뿐입니다. 이를 통해 자신을 더 잘 이해할 수 있을거예요']
    sadness = ['자신에 대한 자비로운 태도를 취하세요. 슬픔은 자신을 이해하고 치유할 기회를 주는 것 입니다.!',
              '힘들 때는 주위의 누군가와 함께하는 것이 도움이 될 수 있어요!',
              '슬픔은 일시적인 감정입니다. 그것은 당신이 지나칠 수 있는 시간이라는 것을 상기하세요!']
    anxiety = ['지금은 어려운 상황일 수 있지만, 우리가 함께 해결할 수 있을거예요!',
              '당신은 충분히 강해요! 이 어려움을 극복할 수 있을거예요!',
              '걱정거리를 하나씩 차분하게 다루어가면, 잘 해결할 수 있을거예요!']
    fear = ['두려움은 새로운 시작을 가로막지만, 그것을 이겨내면 새로운 성장과 기회가 올거예요!',
            '자신의 마음을 대할 용기를 가져주세요. 쉽게 지나치는 일이 아닐 수 있지만, 그것이 새로운 길을 열어줄거예요!',
            '강함은 쉽게 볼 수 없는 용기와 결심에서 나옵니다. 당신은 이미 강하고 용감한 사람입니다!']

    emo_kr_eng = {'기쁨': joy, '당황': confusion, '분노': anger, '슬픔': sadness, '불안': anxiety, '두려움': fear}
    return f"{random.choice(emo_kr_eng[emotion])}"


@app.route('/')  # 메인화면
def index():
    with app.app_context():
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            return render_template('index.html', user=user)
        return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])  # 회원가입
def register():
    with app.app_context():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']

            if is_username_taken(username):
                flash('이미 존재하는 사용자명입니다.', 'danger')
                return redirect(url_for('register'))

            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('계정이 생성되었습니다!', 'success')
            return redirect(url_for('login'))
        return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])  # 로그인화면
def login():
    with app.app_context():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                flash('로그아웃되었습니다!', 'success')
                return redirect(url_for('diary_list'))
            else:
                flash('로그인에 실패했습니다. 다시 시도해주세요.', 'danger')
        return render_template('login.html')

@app.route('/diary_list')
def diary_list():
    with app.app_context():
        if 'user_id' in session:
            
            user = User.query.get(session['user_id'])

            entries = DiaryEntry.query.filter_by(user_id=user.id).order_by(DiaryEntry.date.desc()).all()

            return render_template('diary_list.html', entries=entries)
        else:
            return redirect(url_for('login'))
        
@app.route('/delete_entry/<int:entry_id>', methods=['POST'])
def delete_entry(entry_id):
    with app.app_context():
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            entry = DiaryEntry.query.get(entry_id)

            # 해당 일기를 작성한 사용자인지 확인
            if entry and entry.user_id == user.id:
                db.session.delete(entry)
                db.session.commit()
                return jsonify(success=True)
            else:
                return jsonify(success=False, error="Permission denied")
        else:
            return jsonify(success=False, error="User not logged in")

@app.route('/logout',methods=['POST'])  # 
def logout():
    with app.app_context():
        session.pop('user_id', None)
        return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    with app.app_context():
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            try:
                entry_text = request.json['text']
                entry_date = datetime.strptime(request.json['date'], '%Y-%m-%d')
                # 입력 문장 전처리 및 감정 분류
                def predict(predict_sentence):

                    data = [predict_sentence, '0']
                    dataset_another = [data]

                    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, 64, True, False)
                    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=64, num_workers=0)

                    model.eval()

                    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
                        token_ids = token_ids.long().to(device)
                        segment_ids = segment_ids.long().to(device)
                        valid_length= valid_length

                        out = model(token_ids, valid_length, segment_ids)


                        test_eval=[]
                        for i in out:
                            logits=i
                            logits = logits.detach().cpu().numpy()
                            probabilities = F.softmax(torch.from_numpy(logits), dim=0)

                            if np.argmax(logits) == 0:
                                test_eval.append('"기쁨"이')
                                sentence = "오늘은😊" + test_eval[0] + " 느껴지네요"
                            elif np.argmax(logits) == 1:
                                test_eval.append('"당황"이')
                                sentence = "오늘은😳" + test_eval[0] + " 느껴지네요"
                            elif np.argmax(logits) == 2:
                                test_eval.append('"분노"가')
                                sentence = "오늘은😡" + test_eval[0] + " 느껴지네요"
                            elif np.argmax(logits) == 3:
                                test_eval.append('"불안"이')
                                sentence = "오늘은😨" + test_eval[0] + " 느껴지네요"
                            elif np.argmax(logits) == 4:
                                test_eval.append('"슬픔"이')
                                sentence = "오늘은😭" + test_eval[0] + " 느껴지네요"
                        #확률 logits 
                        #sentence = "오늘은 "  + test_eval[0]  + " 느껴지네요."
                        return sentence, probabilities
                global prediction
                global image_path
                prediction, probability = predict(entry_text)
                prob = probability.tolist()
                image_path = rader_chart(prob,entry_date.strftime('%Y-%m-%d'), str(user.id))
                
                new_entry = DiaryEntry(date=entry_date, text=entry_text, user_id=user.id, analysis_result=prediction, image_path=image_path)
                db.session.add(new_entry)
                db.session.commit()
                print(probability)

                return jsonify(success=True)
            except Exception as e:
                print("Error predicting entry:", str(e))
                return jsonify(success=False, error=str(e))
        else:
            return jsonify(success=False, error="User not logged in")

@app.route('/result')
def result():
    
    return render_template('result.html',prediction = prediction, image_path = image_path)

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/write',methods=['POST'])
def write():
    return render_template('index.html')

@app.route('/recommend_music')
def recommend_music():
    # 감정을 파라미터로 받아오기 (실제로는 사용자의 감정 정보를 받아오는 방식으로 변경)
    emotion = prediction[5:7]

    # 음악 추천 함수 호출
    music_title, music_img_url, music_url = music_recommend(emotion)

    # 추천된 음악 정보를 JSON 형태로 전달
    return jsonify(emotion=emotion, music_title= music_title, music_img_url=music_img_url, music_url=music_url)

@app.route('/recommend_video')
def recommend_video():
    # 감정을 파라미터로 받아오기 (실제로는 사용자의 감정 정보를 받아오는 방식으로 변경)
    emotion = prediction[5:7]

    # 영상 추천 함수 호출
    video_title, video_img_url, video_url = video_recommend(emotion)

    # 추천된 영상 정보를 JSON 형태로 전달
    return jsonify(emotion=emotion, video_title=video_title, video_img_url=video_img_url, video_url=video_url)

@app.route('/get_one_word')
def get_one_word_route():
    emotion = prediction[5:7]
    one_word = get_one_word(emotion)
    return jsonify(one_word=one_word)

if __name__ == '__main__':
    with app.app_context():
        
        db.create_all()
        app.run(debug=True)