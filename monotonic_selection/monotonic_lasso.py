import numpy as np
from itertools import combinations
import pandas as pd
from . import utils
import time
import matplotlib.pyplot as plt

class Lasso:
    def __init__(self, alpha: float=0.001, max_iter: int=1000, tol: float=1e-4, additivity_order: int=1, t_norm='min', log_display=None):
        self.alpha=alpha #L1正則化項のパラメータ
        self.max_iter=max_iter #最大繰り返し回数
        self.tol=tol #収束条件
        self.additivity_order=additivity_order #加法性
        self.t_norm=t_norm #t-ノルム
        self.log_display=log_display #ログの表示
        self.weights=None
        self.intercept_ = None
        self.coef_ = None
        self._name_to_index = None
        self._subset_map = None
        self._superset_map = None
        self._feature_sets = None
        self._feature_names_in_ = None


    #ソフト閾値関数
    @staticmethod
    def soft_thresholding(x, threshold):
        return np.sign(x)*np.maximum(np.abs(x)-threshold,0)
    

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
 

    #下限値の算出
    def _lower_bound(self, weights, current_index):
        lower_bound=-np.inf
        
        set_A_i=self._feature_sets[current_index]

        if len(set_A_i)<2:
            lower_bound=max(lower_bound,0)
        elif len(set_A_i)>=2:
            for _, subset_indices in self._subset_map[current_index].items():
                if subset_indices:
                    m_B_sum=np.sum(weights[subset_indices])
                else:
                    m_B_sum=0.0
                lower_bound=max(lower_bound,-m_B_sum)
        
        for k, element_j in self._superset_map[current_index]:
            m_A=weights[k]
            subset_indices=self._subset_map[k][element_j]
            total_sum=np.sum(weights[subset_indices])
            m_B_sum=total_sum-weights[current_index]
            lower_bound=max(lower_bound,-(m_A+m_B_sum))
        
        return lower_bound
    

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
        X_extended=np.asfortranarray(np.insert(X_extended.values, 0, 1, axis=1))
        return X_extended


    def fit(self,X,y):
        t1=time.time()
        self.weight_change_list = []
        flag=0 #厳密モード管理用フラグ(0:0スキップ，1:厳密モード)
        
        X=self.transform(X)
        variables=self._feature_names_in_
        self._pre_inclusion_map(variables)
        y=y.values
        n_samples, n_features=X.shape
        if self.weights is None or len(self.weights) != n_features:
            self.weights = np.zeros(n_features)
        norm_cols=np.sum(X**2,axis=0)
        residual=y-X@self.weights

        for j in range(self.max_iter):
            old_weight=np.copy(self.weights)

            i=0
            residual_without_i=residual+X[:,i]*self.weights[i]
            new_intercept=np.mean(residual_without_i)

            self.weights[i]=new_intercept
            residual=residual_without_i-X[:,i]*self.weights[i]

            for i in range(1, n_features):
                if norm_cols[i]==0:
                    continue
                residual_without_i=residual+X[:,i]*self.weights[i]
                rho=X[:,i]@residual_without_i
                threshold=n_samples*self.alpha
                weight_now=self.soft_thresholding(rho,threshold)/norm_cols[i]
                
                if flag==0:
                    #0スキップ
                    if weight_now != 0:
                        weight_lower=self._lower_bound(self.weights[1:],i-1)
                        self.weights[i]=max(weight_now,weight_lower)
                    else:
                        self.weights[i]=weight_now
                else:
                    #厳密モード
                    weight_lower=self._lower_bound(self.weights[1:],i-1)
                    self.weights[i]=max(weight_now,weight_lower)
                
                residual=residual_without_i-X[:,i]*self.weights[i]

            weight_change=np.linalg.norm(self.weights-old_weight)
            self.weight_change_list.append(weight_change)
            if self.log_display==True:
                print(f"[{j+1}/{self.max_iter}], weight_change: {weight_change}")
            if weight_change<self.tol:
                if self.monotonic_check(self.weights[1:])==0:
                    t2=time.time()
                    print(f"converged!\ntime[s]: {t2-t1:.2f}")
                    break
                else:
                    #厳密モードに移行
                    flag=1
        self.intercept_=self.weights[0]
        self.coef_=self.weights[1:]


    def predict(self, X):
        X=self.transform(X)
        return X@self.weights
    

    def r2_score(self, X, y):
        y_true=y.values
        y_pred=self.predict(X)
        ss_total=np.sum((y_true-np.mean(y_true))**2)
        ss_residual=np.sum((y_true-y_pred)**2)
        return 1-(ss_residual/ss_total)
    

    #収束状況の図示
    def plot_weight_change(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.weight_change_list, lw=3, c='b')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        #plt.legend(fontsize=14)
        plt.show()
    

    #選択された変数の出力
    def selected_variables(self):
        title="selected variables"
        print(title.center(60,"-"))
        for i in range(len(self.coef_)):
            if self.coef_[i]!=0:
                print(f"{self._feature_names_in_[i]}: {self.coef_[i]}")
        print(f"selected variables: {np.count_nonzero(self.coef_)}/{len(self._feature_names_in_)}")
        print("-"*60)
    

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
