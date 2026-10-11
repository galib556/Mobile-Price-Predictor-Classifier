
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingRegressor,VotingClassifier,StackingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
import pickle
warnings.filterwarnings('ignore')
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.corr()['price_range'].sort_values(ascending=False)
train.drop(['blue','clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc','sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi'],inplace=True,axis=1)





X=train.drop('price_range',axis=1)
y=train['price_range']
cat_cols=[]
num_cols=["battery_power","px_height","px_width","ram"]
num_cols_transform=Pipeline([
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])
cat_cols_transform=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])
pipe=ColumnTransformer([
    ('num',num_cols_transform,num_cols),
    ('cat',cat_cols_transform,cat_cols)
])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr=LogisticRegression()
base=DecisionTreeClassifier(max_depth=3,random_state=42)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
ada=AdaBoostClassifier(estimator=base)
gb=GradientBoostingClassifier(n_estimators=100,random_state=42)
xgb=XGBClassifier(n_estimators=100,random_state=42)
svc=SVC()
knn=KNeighborsClassifier()
voting_cl=VotingClassifier(estimators=[('lr',lr),('rf',rf),('ada',ada),('gb',gb),('xgb',xgb),('dt',base),('svc',svc),('knn',knn)],voting='hard')
stacking_cl=StackingClassifier(estimators=[('lr',lr),('rf',rf),('ada',ada),('gb',gb),('xgb',xgb),('dt',base),('svc',svc),('knn',knn)],final_estimator=LogisticRegression())



model_to_train={
    'lr':LogisticRegression(),
    'rf':RandomForestClassifier(),
    'ada':AdaBoostClassifier(),
    'gb':GradientBoostingClassifier(),
    'xgb':XGBClassifier(),
    'svc':SVC(),
    'knn':KNeighborsClassifier(),
    'voting_cl':voting_cl,
    'stacking_cl':stacking_cl
}
result=[]
for model_name,model in model_to_train.items():
    pipe_model=Pipeline([
        ('preprocess',pipe),
        ('model',model)
    ])
    pipe_model.fit(X_train,y_train)
    y_pred=pipe_model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    mae=mean_absolute_error(y_test,y_pred)
    result.append({
        'model':model_name,
        'r2':r2,
        'accuracy':accuracy,
        'rmse':rmse,
        'mae':mae
    })
result=pd.DataFrame(result).sort_values(by='r2',ascending=False)
print(result)


best_model=result.iloc[0]['model']
best_model_object=model_to_train[best_model]
final_pipe=Pipeline([
    ('preprocess',pipe),
    ('model',best_model_object)
])
final_pipe.fit(X_train,y_train)
y_pred=final_pipe.predict(X_test)


lr_pipe=Pipeline([
    ('preprocess',pipe),
    ('model',LogisticRegression())
])
lr_pipe.fit(X_train,y_train)
y_pred=lr_pipe.predict(X_test)


cv_scores=cross_val_score(lr_pipe,X_train,y_train,cv=5,scoring='f1_macro')
print(cv_scores)
print(np.mean(cv_scores))

param_grid={
    'model__C':[0.001,0.01,0.1,1,10,100]
}


grid_search=GridSearchCV(lr_pipe,param_grid,cv=5,scoring='f1_macro')
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

lr_pipe.fit(X_train,y_train)
imp=permutation_importance(lr_pipe,X_test,y_test,n_repeats=100,random_state=42,n_jobs=2,scoring='f1_macro')
imp=pd.Series(imp.importances_mean,index=X_train.columns).sort_values(ascending=False)
print(imp)
features=['battery_power','px_height','px_width','ram']
pipe4=Pipeline([
    ('Scaler',StandardScaler()),
    ('model',LogisticRegression(C=grid_search.best_params_['model__C'],max_iter=2000))
])
pipe4.fit(X_train[features],y_train)
y_pred=pipe4.predict(X_test[features])
pickle.dump(pipe4,open('lr_model.pkl','wb'))