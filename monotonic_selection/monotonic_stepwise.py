import numpy as np
from sklearn import linear_model
import pandas as pd
import time
from itertools import combinations
from . import utils
import matplotlib.pyplot as plt

class MonoStepwise:
    def __init__(self, criterion='f_value', max_iter: int=1000, additivity_order: int=1, t_norm='min'):
        self.criterion=criterion #判断基準
        self.max_iter=max_iter #最大繰り返し回数
        self.additivity_order=additivity_order #加法性加法性
        self.t_norm=t_norm #t-ノルム
        self._selected_variables = []
        self._name_to_index = None
        self._subset_map = None
        self._superset_map = None
        self.intercept_ = None
        self.coef_ = None
        self.model = None

    
    #集合間の包含関係の記憶
    def _pre_inclusion_map(self, variables):
        self._name_to_index={name: i for i, name in enumerate(variables)}
        self._feature_sets=[set(name.split('-')) for name in variables]

        self._subset_map=[{} for _ in range(len(variables))]
        for i, set_A in enumerate(self._feature_sets):
            for element_j in set_A:
                subset_indices=[]
                for k, set_B in enumerate(self._feature_sets):
                    if i==k: 
                        continue
                    if element_j in set_B and set_B.issubset(set_A) and set_A!=set_B:
                        subset_indices.append(k)
                self._subset_map[i][element_j]=subset_indices
        
        self._superset_map=[[] for _ in range(len(variables))]
        for i, set_B in enumerate(self._feature_sets):
            for element_j in set_B:
                for k, set_A in enumerate(self._feature_sets):
                    if i==k or len(set_A)<2:
                        continue
                    if set_B.issubset(set_A):
                        self._superset_map[i].append((k,element_j))
 

    #単調性の確認
    def monotonic_check(self, weights):
        epsilon=1e-15
        for i in range(len(weights)):
            for _, subset_indices in self._subset_map[i].items():
                if subset_indices:
                    m_B=np.sum(weights[subset_indices])
                else:
                    m_B=0
                if weights[i]<-m_B-epsilon:
                    return -1
        return 0
    

    #残差平方和
    def rss(self, a: pd.Series, b: pd.Series) -> float:
        return ((a-b)**2).sum()
    

    #選択された変数でモデルを学習して，予測値とモデルを返す
    def fit_and_predict(self, X, y, features):
        if not features:
            pred=np.full(y.shape[0],y.mean())
            return None, pred
        
        model=linear_model.LinearRegression()
        model.fit(X[features],y)
        pred=model.predict(X[features])
        return model, pred
    

    #AIC(赤池情報量基準)
    def aic(self, rss, n, k):
        return n*np.log(rss/n)+2*k
    

    #BIC(ベイズ情報量基準)
    def bic(self, rss, n, k):
        return n*np.log(rss/n)+k*np.log(n)
    

    def transform(self, X):
        X_extended=X.copy()
        variables=X.columns.tolist()
        new_col={}

        if self.additivity_order!=1:
            for r in range(2,self.additivity_order+1):
                for comb in combinations(variables, r):
                    col_name = '-'.join(comb)
                    temp = X[comb[0]]
                    for i in comb[1:]:
                        temp = getattr(utils,self.t_norm)(temp,X[i])
                    new_col[col_name] = temp

        X_extended=pd.concat([X_extended,pd.DataFrame(new_col)],axis=1)
        variables+=list(new_col.keys())

        self._feature_names_in_=variables
        return X_extended


    #変数選択ステップ
    def forward_selection(self, X, y):
        variables=X.columns.tolist()
        while True:
            best_f_value=3.84 #分散比Fの閾値でF分布の有意水準5%に相当する
            best_v=None
            _, prediction_old=self.fit_and_predict(X,y,self._selected_variables)
            m=y.shape[0]
            l=len(self._selected_variables)
            rss_old=self.rss(y,prediction_old)
            best_aic=self.aic(rss_old,m,l)
            best_bic=self.bic(rss_old,m,l)
            for i in variables:
                if i not in self._selected_variables:
                    new_selected_variables=self._selected_variables.copy()
                    new_selected_variables.append(i)
                    model_new, prediction_new=self.fit_and_predict(X,y,new_selected_variables)
                    rss_new=self.rss(y,prediction_new)
                    k=len(new_selected_variables)
                    if self.criterion=='f_value':
                        dispersion_ratio=((rss_old-rss_new)/(k-l))/(rss_new/(m-k-1))
                    elif self.criterion=='aic':
                        aic_new=self.aic(rss_new,m,k)
                    elif self.criterion=='bic':
                        bic_new=self.bic(rss_new,m,k)
                    weights=np.zeros(len(variables))
                    for idx, name in enumerate(new_selected_variables):
                        weights[self._name_to_index[name]]=model_new.coef_[idx]
                    if self.criterion=='f_value':
                        if dispersion_ratio > best_f_value:
                            if self.monotonic_check(weights)==0:
                                best_f_value=dispersion_ratio
                                best_v=i
                    elif self.criterion=='aic':
                        if aic_new < best_aic:
                            if self.monotonic_check(weights)==0:
                                best_aic=aic_new
                                best_v=i
                    elif self.criterion=='bic':
                        if bic_new < best_bic:
                            if self.monotonic_check(weights)==0:
                                best_bic=bic_new
                                best_v=i
            if best_v==None:
                break
            self._selected_variables.append(best_v)
            print(f"selection: {best_v}")
        return 0


    #変数除去ステップ
    def backward_elimination(self, X, y):
        variables=X.columns.tolist()
        while True:
            worst_f_value=2.71 #分散比Fの閾値でF分布の有意水準10%に相当する
            worst_v=None
            _, prediction_old=self.fit_and_predict(X,y,self._selected_variables)
            m=y.shape[0]
            l=len(self._selected_variables)
            rss_old=self.rss(y,prediction_old)
            worst_aic=self.aic(rss_old,m,l)
            worst_bic=self.bic(rss_old,m,l)
            for i in self._selected_variables:
                new_selected_variables=self._selected_variables.copy()
                new_selected_variables.remove(i)
                model_new, prediction_new=self.fit_and_predict(X,y,new_selected_variables)
                rss_new=self.rss(y,prediction_new)
                k=len(new_selected_variables)
                if self.criterion=='f_value':
                    dispersion_ratio=((rss_old-rss_new)/(k-l))/(rss_new/(m-k-1))
                elif self.criterion=='aic':
                    aic_new=self.aic(rss_new,m,k)
                elif self.criterion=='bic':
                    bic_new=self.bic(rss_new,m,k)
                weights=np.zeros(len(variables))
                for idx, name in enumerate(new_selected_variables):
                    weights[self._name_to_index[name]]=model_new.coef_[idx]
                if self.criterion=='f_value':
                    if dispersion_ratio < worst_f_value:
                        if self.monotonic_check(weights)==0:
                            worst_f_value=dispersion_ratio
                            worst_v=i
                elif self.criterion=='aic':
                    if aic_new < worst_aic:
                        if self.monotonic_check(weights)==0:
                            worst_aic=aic_new
                            worst_v=i
                elif self.criterion=='bic':
                    if bic_new < worst_bic:
                        if self.monotonic_check(weights)==0:
                            worst_bic=bic_new
                            worst_v=i
            if worst_v==None:
                break
            self._selected_variables.remove(worst_v)
            print(f"elimination: {worst_v}")
        return 0

    
    def fit(self, X, y):
        t1=time.time()
        self.weight_change_list = []
        X=self.transform(X)
        variables=self._feature_names_in_
        self._pre_inclusion_map(variables)

        for count in range(self.max_iter):
            forward_old=len(self._selected_variables)
            self.forward_selection(X,y)
            forward_new=len(self._selected_variables)
            if forward_old == forward_new:
                break
            self.backward_elimination(X,y)
            backward_old=forward_new
            backward_new=len(self._selected_variables)
            if backward_old == backward_new:
                break

        self.model, _=self.fit_and_predict(X,y,self._selected_variables)
        weights=np.zeros(len(variables))
        for idx, name in enumerate(self._selected_variables):
            weights[self._name_to_index[name]]=self.model.coef_[idx]
        self.intercept_=self.model.intercept_
        self.coef_=weights
        t2=time.time()
        print(f"converged!\ntime[s]: {t2-t1:.2f}")


    def predict(self, X):
        X=self.transform(X)
        if self.model is None or not self._selected_variables:
            return np.full(X.shape[0], self.intercept_)
        
        return self.model.predict(X[self._selected_variables])
    
    
    def r2_score(self, X, y):
        X=self.transform(X)
        if self.model is None or not self._selected_variables:
            return 0.0
        
        return self.model.score(X[self._selected_variables],y)


    #選択された変数の出力
    def selected_variables(self):
        selected_variables_set=[]
        title="selected variables"
        print(title.center(60,"-"))
        for i in range(len(self.coef_)):
            if self.coef_[i]!=0:
                print(f"{self._feature_names_in_[i]}: {self.coef_[i]}")
                selected_variables_set.append(self._feature_names_in_[i])
        print(f"selected variables: {np.count_nonzero(self.coef_)}/{len(self._feature_names_in_)}")
        print("-"*60)
        return selected_variables_set
    

    #ファジィ測度の出力
    def fuzzy(self):
        self.fuzzy_measures=None

        n_features=len(self._feature_sets)
        fuzzy_matrix=np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if self._feature_sets[j].issubset(self._feature_sets[i]):
                    fuzzy_matrix[i,j]=1

        self.fuzzy_measures = fuzzy_matrix @ self.coef_

        title="fuzzy"
        print(title.center(60,"-"))
        for name, measure in zip(self._feature_names_in_, self.fuzzy_measures):
            print(f"{name}: {measure}")
        print("-"*60)


    #相互作用指標の出力
    def interaction(self):
        self.interaction_values=None

        lengths=[len(s) for s in self._feature_sets]
        n_features=len(self._feature_sets)

        interaction_matrix=np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if self._feature_sets[i].issubset(self._feature_sets[j]):
                    denominator=lengths[j]-lengths[i]+1
                    interaction_matrix[i,j]=1/denominator

        self.interaction_values=interaction_matrix @ self.coef_

        title="interaction"
        print(title.center(60,"-"))
        for name, val in zip(self._feature_names_in_, self.interaction_values):
            print(f"{name}: {val}")
        print("-"*60)
    

    #シャプレー値の出力
    def shapley(self):
        all_elements=sorted(list(set().union(*self._feature_sets)))
        n_elements=len(all_elements)
        n_subsets=len(self._feature_names_in_)

        shapley_matrix=np.zeros((n_elements,n_subsets))

        for i, element in enumerate(all_elements):
            for j, subset in enumerate(self._feature_sets):
                if element in subset:
                    shapley_matrix[i, j]=1/len(subset)
        
        shapley_vals=shapley_matrix @ self.coef_

        title="shapley"
        print(title.center(60,"-"))
        for name, val in zip(all_elements, shapley_vals):
            print(f"{name}: {val}")
        print("-"*60)

        return all_elements, shapley_vals


    #シャプレー値の図示
    def shapley_plot(self):
        labels,shaplay_vals=self.shapley()
        plt.figure(figsize=(10,6))
        plt.barh(labels,shaplay_vals,color='skyblue')
        plt.xlabel('Shapley value')
        plt.ylabel('Columns')
        plt.title('Shapley Value')
        plt.gca().invert_yaxis() 
        plt.show()
        





