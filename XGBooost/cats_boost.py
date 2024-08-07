import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV , learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# 데이터 로드 및 전처리
non_flood_data_path = '침수가아닌날데이터2022_2020.csv'
flood_data_path = '침수학습데이터.csv'

non_flood_data = pd.read_csv(non_flood_data_path)
flood_data = pd.read_csv(flood_data_path)

non_flood_data['target'] = 0
flood_data['target'] = 1

combined_data = pd.concat([non_flood_data, flood_data], ignore_index=True)

selected_columns = [
    '배수정보', '침수된 지역의 평균 지형 고도', '1시간 누적 강수량(mm)', '1시간 10분 평균 강수량(mm)', '1시간 시간당 강수량(mm)',
    '2시간 누적 강수량(mm)', '2시간 10분 평균 강수량(mm)', '2시간 시간당 강수량(mm)', '3시간 누적 강수량(mm)', '3시간 10분 평균 강수량(mm)',
    '3시간 시간당 강수량(mm)', '4시간 누적 강수량(mm)', '4시간 10분 평균 강수량(mm)', '4시간 시간당 강수량(mm)', '5시간 누적 강수량(mm)',
    '5시간 10분 평균 강수량(mm)', '5시간 시간당 강수량(mm)', '6시간 누적 강수량(mm)', '6시간 10분 평균 강수량(mm)', '6시간 시간당 강수량(mm)',
    '7시간 누적 강수량(mm)', '7시간 10분 평균 강수량(mm)', '7시간 시간당 강수량(mm)', '8시간 누적 강수량(mm)', '8시간 10분 평균 강수량(mm)',
    '8시간 시간당 강수량(mm)', '9시간 누적 강수량(mm)', '9시간 10분 평균 강수량(mm)', '9시간 시간당 강수량(mm)', '10시간 누적 강수량(mm)',
    '10시간 10분 평균 강수량(mm)', '10시간 시간당 강수량(mm)', '11시간 누적 강수량(mm)', '11시간 10분 평균 강수량(mm)', '11시간 시간당 강수량(mm)',
    '12시간 누적 강수량(mm)', '12시간 10분 평균 강수량(mm)', '12시간 시간당 강수량(mm)'
]

X = combined_data[selected_columns]
y = combined_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 하이퍼파라미터 범위 설정
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5]
}

# CatBoost 분류기 초기화
cat_model = cb.CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    silent=True
)

# GridSearchCV 초기화
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# GridSearchCV 수행
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터와 성능
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-validation Accuracy:", grid_search.best_score_)

# 최적의 모델로 테스트 데이터 예측 및 평가
best_model = grid_search.best_estimator_

# 예측 확률을 계산
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# 임계값 설정
threshold = 0.9  # 예: 임계값을 0.6으로 설정

# 임계값에 따라 0과 1로 분류
y_pred_threshold = (y_pred_prob >= threshold).astype(int)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred_threshold)
report = classification_report(y_test, y_pred_threshold)

print(f'Accuracy with threshold {threshold}: {accuracy}')
print(f'Classification Report:\n{report}')

# Confusion Matrix 시각화
conf_matrix = confusion_matrix(y_test, y_pred_threshold)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Flood', 'Flood'], yticklabels=['Non-Flood', 'Flood'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold: {threshold})')
plt.show()

# 피처 중요도 시각화
feature_importances = best_model.get_feature_importance()
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 12))
plt.barh(range(len(feature_importances)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(feature_importances)), [selected_columns[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance (CatBoost)')
plt.show()


# 학습 곡선 (Learning Curve)
train_sizes, train_scores, test_scores = learning_curve(
    cat_model.set_params(**grid_search.best_params_), 
    X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.title('Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.show()



# 모델 저장
model_filename = 'catboost_best_model.pkl'
joblib.dump(best_model, model_filename)
print(f'Model saved to {model_filename}')

# 모델 불러오기
loaded_model = joblib.load(model_filename)

# 불러온 모델로 예측 수행 (테스트 데이터)
y_loaded_pred = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, y_loaded_pred)
print(f'Loaded model accuracy: {loaded_accuracy}')