# 文件路径: app/train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ---- 数据部分 (保持不变) ----
data = {
    'text': [
        # --- 10个文本元素 ---
        'this movie was fantastic, I loved it', # 1
        'what a horrible film, a complete waste of time', # 2
        'the acting was superb and the plot was gripping', # 3
        'I would not recommend this to anyone, it was boring', # 4
        'an amazing and wonderful experience', # 5
        'the director is a genius', # 6
        'I fell asleep halfway through', # 7
        'This movie was just okay, not great but not terrible either.', # 8
        'The visual effects were incredible, even if the story was weak.', # 9
        'A decent film for a Tuesday night.' # 10
    ],
    'sentiment': [
        # --- 同样是10个标签元素 ---
        'positive', # 1
        'negative', # 2
        'positive', # 3
        'negative', # 4
        'positive', # 5
        'positive', # 6
        'negative', # 7
        'negative', # 8
        'positive', # 9
        'positive'  # 10
    ]
}
df = pd.DataFrame(data)

X = df['text']
y = df['sentiment']

# ---- 训练部分 (保持不变) ----
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X_vectorized, y)

# ---- 保存部分 (核心修改) ----
# 我们希望将模型文件保存在项目的根目录，而不是app/目录里
# 这样可以方便Dockerfile和app.py加载
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True) # 创建models文件夹

joblib.dump(model, os.path.join(MODEL_DIR, 'sentiment_model.joblib'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'sentiment_vectorizer.joblib'))

print(f"模型和特征提取器已成功保存到 {MODEL_DIR} 文件夹中。")