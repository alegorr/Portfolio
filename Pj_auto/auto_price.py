#!/usr/bin/env python
# coding: utf-8

# 
# Привет очередной раз, меня зовут Люман Аблаев. Сегодня я проверю твой проект.
# <br> Дальнейшее общение будет происходить на "ты" если это не вызывает никаких проблем.
# <br> Желательно реагировать на красные комментарии ('исправил', 'не понятно как исправить ошибку', ...)
# <br> Пожалуйста, не удаляй комментарии ревьюера, так как они повышают качество повторного ревью.
# 
# Комментарии будут в <font color='green'>зеленой</font>, <font color='blue'>синей</font> или <font color='red'>красной</font> рамках:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Успех:</b> Если все сделано отлично
# </div>
# 
# <div class="alert alert-block alert-info">
# <b>Совет: </b> Если можно немного улучшить
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Ошибка:</b> Если требуются исправления. Работа не может быть принята с красными комментариями.
# </div>
# 
# -------------------
# 
# Будет очень хорошо, если ты будешь помечать свои действия следующим образом:
# <div class="alert alert-block alert-warning">
# <b>Комментарий студента:</b> ...
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Изменения:</b> Были внесены следующие изменения ...
# </div>
# 
# 
# 
# 
# 
# 
# 
# <font color='orange' style='font-size:24px; font-weight:bold'>Общее впечатление</font>
# * Приятно было снова проверять твою работу
# - Я постарался оставить полезные советы, надеюсь они тебе пригодятся.
# - Увы, я обнаружил небольшие недочеты  в работе, но я думаю  у тебя не займет много усилий их исправить.
# - Давай еще разок

# 
# <font color='orange' style='font-size:24px; font-weight:bold'>Общее впечатление[2]</font>
# * Спасибо за усердность!
# - Было приятно с тобой сотрудничать.
# - Недочеты исправлены - работа полностью корректна
# - Не буду больше задерживать, продолжай в том же духе.
# 

# # Определение стоимости автомобилей

# Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 
# 
# Заказчику важны:
# 
# - качество предсказания;
# - скорость предсказания;
# - время обучения.

# ## Подготовка данных

# In[ ]:


pip install lightgbm


# In[ ]:


pip install category_encoders


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm
import category_encoders as ce

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    make_scorer
)


# 
# <div class="alert alert-block alert-success">
#     
# <b>Успех:</b> Импорты на месте
# </div>

# In[ ]:


data_old = pd.read_csv('/datasets/autos.csv')


# In[ ]:


display(data_old.head())
data_old.info()
print('')
print('Количество пропусков:', data_old.isna().sum())
display(pd.DataFrame(round(data_old.isna().mean().sort_values(ascending=False)*100,1)).style.background_gradient('coolwarm'))
print('Количество явных дубликатов:', data_old.duplicated().sum())
plt.figure(figsize = (15,5))
sns.heatmap(data_old.corr(), annot = True)
plt.title('Корреляция между признаками')
plt.show()


# Всего 15 колонок и 354369 строк. 
# 
# 1) Признаки: DateCrawled, DateCreated, LastSeen относятся к типу данных object, хотя это дата и время;
# 
# 2) Пропущенные значения: Repaired (20%), VehicleType (10%), FuelType (9%), Gearbox (5%), Model (5%);
# 
# 3) Количество явных дубликатов: 4;
# 
# 4) Корреляции между признаками нет, кроме слабой - между ценой и пробегом.

# 
# <div class="alert alert-block alert-success">
#     
# <b>Успех:</b> Первичный осмотр проведен
# </div>

# In[ ]:


data_repaired = data_old.query('Repaired == "yes"')

data_repaired.head()


# In[ ]:


data_repaired[['RegistrationYear', 'Kilometer']].describe()


# In[ ]:


data_old['Kilometer'].unique()


# Большинство автомобилей, которые имели ремонт, были выпущены до 2006 г. и с пробегом более 150000 км. Исходя из этого, сделаем допущение, что пропущенные значения для автомобилей с таким же пробегом и возрастом, соответствуют наличию ремонта. 

# In[ ]:


data_old.loc[(data_old['RegistrationYear'] <= 2005) 
         & (data_old['Kilometer'] >= 150000), 'Repaired'] = data_old.loc[(data_old['RegistrationYear'] <= 2005) & 
                                                                 (data_old['Kilometer'] >= 150000), 'Repaired'].fillna('yes')


# In[ ]:


display(pd.DataFrame(round(data_old.isna().mean().sort_values(ascending=False)*100,1)).style.background_gradient('coolwarm'))


# In[ ]:


display(data_old['DateCrawled'].min())
data_old['DateCrawled'].max()


# Удалим признаки, которые не будут влиять на стоимость автомобиля или будут, но незначимо. 

# In[ ]:


data = data_old.drop(['NumberOfPictures', 'RegistrationMonth', 'PostalCode', 'DateCrawled', 'DateCreated', 'LastSeen'], axis=1)


# <div class="alert alert-block alert-success">
# <b>Успех:</b> Согласен с перечнем неинформативных колонок
# </div>

# Удалим строки, где нет указаний на модель, так как восстановить ее будет сложно, а признак, вероятно, будет определять стоимость. 

# In[ ]:


data = data.dropna(subset=['Model'], axis=0)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# Из таблицы видно, что в столбцах: цена, год регистрации, мощность могут находиться аномальные значения. 

# In[ ]:


data.boxplot('Price', vert=False, figsize=(15, 3))
plt.xlabel('Price')
plt.title('Price')
plt.show()


# In[ ]:


data.query('Price < 100')


# 0 в качестве цены указан у большинства значений, еще примерно у 1000 цена низкая. Конечно, можно подумать, что продавец хочет просто избавится от машины, но формальная логика говорит, что стоимость автомобиля как металлолома будет больше. Сделаем границу в 100 евро. 

# In[ ]:


data = data.drop(data.query('Price < 100').index)


# <div class="alert alert-block alert-success">
# <b>Успех:</b>   Самое интересное, что на нескольких сайтов по продажам подержанных автомобилей в Германии (страну можно определить по почтовым индексам, большинство из них из Германии) показало, что цены начинаются действительно с 0 и 1 евро, но это единичные объявления и не понятно, то ли это ошибка при заполнении формы, то ли машины на металлолом. Но я  все-таки склоняюсь к тому, чтобы избавиться от таких данных
# </div>
# 

# In[ ]:


data['RegistrationYear'].hist(bins = 100)


# In[ ]:


data.query('RegistrationYear > 2016 or RegistrationYear < 1950')


# Указаны колясницы и машины будущего. Последняя дата скачивания анкеты из базы 2016-04-07 14:36:58, но пристутствуют автомобили от 2017 г., что составляет 11844 наблюдений. Интерпретировать данные изменения лучше всего с руководителями сервиса  «Не бит, не крашен», возможно, что дата ошибочная. Скачали базу после обновления, например. Так как точная дата обращения заказчика неизвестна, а количество ошибочных записей большое, чтобы думать, что это несистемная ошибка, удалю данные актуальные на настоящий момент.  

# In[ ]:


data = data.drop(data.query('RegistrationYear > 2023 or RegistrationYear < 1950').index)


# 
# <div class="alert alert-block alert-info">
# <b>Совет:</b> Советую посмотреть на дату выгрузки анкет - там тоже можно увидеть полезную информацию по поводу границ года регистраци.
# </div>

# <div class="alert alert-block alert-warning">
# <b>Изменения:</b> Посмотрел дату выгрузки анкет, изменил умозаключение, которое было, но код оставил прежним. 
# </div>

# 
# <div class="alert alert-block alert-success">
# <b>Успех[2]:</b> Ок
# </div>

# In[ ]:


data['RegistrationYear'].hist(bins = 50)


# In[ ]:


data['Power'].hist(bins = 50)


# In[ ]:


data.query('Power > 500 or Power < 0.75')


# In[ ]:


data.query('Power == 0')['Price'].hist(bins = 100)


# In[ ]:


data.loc[(data['Power'] > 500) | ((data['Power'] < 0.75) & (data['Power'] != 0)),'Power']


# 366 машин имеют мощность, которая не соответствует раельности. Заменим значения на пропуски, чтоб потом их можно было заполнить.

# In[ ]:


data.loc[(data['Power'] > 500) | ((data['Power'] < 0.75) & (data['Power'] != 0)),'Power'] = np.nan


# 
# 
# <div class="alert alert-block alert-info">
# <b>Совет:</b>  Вопрос на подумать: Как думаешь а машины с 0-ой мощностью могут быть просто без двигателя?
# </div>

# <div class="alert alert-block alert-warning">
# <b>Изменения:</b> Веротяно, да. Оставил машины с 0 мощностью без изменений.
# </div>

# 
# <div class="alert alert-block alert-success">
# <b>Успех[2]:</b> Но здесь я бы не спешил. Нужно подумать нужны ли нашему сервису такие автомобили в качестве примеров?
# </div>

# In[ ]:


missing_values = data['Power'].isnull().sum()
print("Количество пропущенных значений в столбце Power:", missing_values)


# In[ ]:


data['Power'] = data.groupby(['Model', 'Brand'])['Power'].transform(lambda x: x.fillna(x.median()) if x.notnull().any() else x)


# In[ ]:


missing_values = data['Power'].isnull().sum()
print("Количество пропущенных значений в столбце Power:", missing_values)


# In[ ]:


data['Power'].hist(bins = 100)


# In[ ]:


data.boxplot('Power', vert = False, figsize = (15,3))


# In[ ]:


data.loc[data['Power'] == 0,'Power']


# In[ ]:


repaired_counts = data.loc[data['Power'] == 0, 'Repaired'].value_counts()
repaired_counts.plot(kind='pie')
plt.show()


# In[ ]:


print('Медиана стоимости автомобиля без двигателя:', data.loc[data['Power'] == 0,'Price'].median())
print('Медиана стоимости автомобиля с двигателем:', data.loc[data['Power'] != 0,'Price'].median())


# Если решить, что мощность, равная 0, соответствует машинам без двигателя, то смущает такое большое количество (более 30321). Стоит посоветоваться с экспертами рынка, чтобы узнать, является ли это аномалией. Половина этих машин не подвергалась ремонту, то есть, как только сгорал двигатель, то машину сразу стремились продать. Тоже вопрос, соответствует ли это реальной жизни...Но судя по медиане стоимости, машины без двигателя в среднем стоят в 2 раза меньше, что похоже на правду.  

# In[ ]:


display(pd.DataFrame(round(data.isna().mean().sort_values(ascending=False)*100,1)).style.background_gradient('coolwarm'))


# Тип коробки передач, вид топлива, тип кузова можно примерно восстановить, зная модель и марку автомобиля.

# In[ ]:


data['Gearbox'] = data.groupby(['Model', 'Brand'])['Gearbox'].transform(lambda x: x.fillna(x.mode().iloc[0]) if x.notnull().any() else x)


# In[ ]:


#data['FuelType'] = data.groupby(['Model', 'Brand'])['FuelType'].transform(lambda x: x.fillna(x.mode().iloc[0]))

data['FuelType'] = data.groupby(['Model', 'Brand', 'Gearbox'])['FuelType'].transform(lambda x: x.fillna(x.mode().iloc[0]) if x.notnull().any() else x)


# In[ ]:


data['Repaired'] = data.groupby(['Model', 'Brand', 'Gearbox', 'FuelType'])['Repaired'].transform(lambda x: x.fillna(x.mode().iloc[0]) if x.notnull().any() else x)


# In[ ]:


data.loc[data['Model'] == 'golf']['VehicleType'].unique()


# In[ ]:


data['VehicleType'] = data.groupby(['Model', 'Brand', 'Gearbox', 'FuelType'])['VehicleType'].transform(lambda x: x.fillna(x.mode().iloc[0]) if x.notnull().any() else x)


# In[ ]:


display(pd.DataFrame(round(data.isna().mean().sort_values(ascending=False)*100,1)).style.background_gradient('coolwarm'))


# In[ ]:


data['FuelType'].unique()


# - petrol', 'gasoline' - вид топлива бензин;
# - 'lpg', 'cng' - - вид топлива газ.
# 
# Объединим их.

# In[ ]:


data.loc[data['FuelType'] == 'petrol', 'FuelType'] = 'gasoline'
data.loc[data['FuelType'] == 'lpg', 'FuelType'] = 'cng'


# In[ ]:


data['FuelType'].unique()


# <div class="alert alert-block alert-success">
# <b>Успех:</b>  Пропуски обработаны хорошим образом
# </div>
# 
# 
# <div class="alert alert-block alert-info">
# <b>Совет:</b> У fuel_type есть категории, которые означают одно и тоже - их можно объединить, либо подумать может они действительно означают, что-то разное.
# </div>
# 

# <div class="alert alert-block alert-warning">
# <b>Изменения:</b> Поправил fuel_type.
# </div>

# 
# <div class="alert alert-block alert-info">
# <b>Совет[2]:</b> Насчет бензина согласен, а вот газ, как по мне, имеет важные различия, поэтому газ лучше не объединять
# </div>

# In[ ]:


data = data.dropna()


# **Промежуточный вывод**
# 
# Были предоставлены данные, которые имели 15 колонок и 354369 строк.
# 
# 1) Признаки: DateCrawled, DateCreated, LastSeen относились к типу данных object, хотя это дата и время;
# 
# 2) Пропущенные значения были в колонках: Repaired (20%), VehicleType (10%), FuelType (9%), Gearbox (5%), Model (5%);
# 
# 3) Количество явных дубликатов было: 4;
# 
# 4) Корреляции между признаками не было, кроме слабой - между пробегом и целевым признаком ценой.
# 
# 5) В столбцах: цена, год регистрации, мощность находились аномальные значения.
# 
# Столбцы NumberOfPictures, RegistrationMonth, PostalCode, DateCrawled, DateCreated, LastSeen были удалены, так как их значение на стоимость автомобиля, вероятно, незначимо. Также были удаления строки, где не была указана модель, так как восстановить признак было бы невозможно логическим путем. 10642 наблюдений имели 0 в качестве цены, еще около 1000 цена была очень низкая. Учитывая, что это целевой признак, были сохранены только наблюдения с ценой выше 99 евро. Также были удалены наблюдения с годом регистрации автомобиля позже 2023 и раньше 1950 гг. 
# 
# Обратить внимание. Последняя дата скачивания анкеты из базы 2016-04-07 14:36:58, но пристутствуют автомобили от 2017 г., что составляет 11844 наблюдений. Интерпретировать данные изменения лучше всего с руководителями сервиса «Не бит, не крашен», возможно, что дата ошибочная. Скачали базу после обновления, например. Поэтому было принято решения их сохранить на данный момент.
# 
# Часть пропусков в столбце Repaired (автомобили, которые имели ремонт), были заполнены с использованием допущения, что большинство машин, согласно заполненным данным, выпущенных до 2006 г. и с пробегом более 150000 км. имели ремонт. 366 значений в столбце Power (мощность) имели значения, которые мало соответствовали раельности. Они были заменены на медианные значения мощности в соответствии с моделью и брендом. VehicleType (тип кузова), FuelType (вид топлива), Gearbox (тип коробки передач), часть Repaired (ремонт) были приблизетельно восстановлены по моде (самым частым значениям) для своей модель и марки. В столбце вид топлива petrol и gasoline (бензин) и lpg и cng (газ) были объединены.

# <div class="alert alert-block alert-success">
# <b>Успех:</b> В целом хорошая, детальная предобработка - идем дальше
# </div>

# ### Поработаем с признаками

# In[ ]:


target = data['Price']
features = data.drop(['Price'], axis = 1)


# In[ ]:


#features = pd.get_dummies(features_old, columns=['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'Repaired'], drop_first=True)


# 
# <div class="alert alert-block alert-danger">
# 
# <b>Ошибка:</b>  OHE правильный выбор для линейных моделей. Но для всех других моделей выбор плохой (из-за фактора модели, который порождает много факторов), для заказчика важно время обучения и скорость предсказания, а данные модели могут работать и с другими методами кодировками значительно быстреее, не теряя при этом в качестве.
# 
# Я тебе могу предложить, что можно сделать:
#     
# - Сделать 1 набор данных: 
#     - Закодировать для всех моделей методом TargetEncoder, BinaryEncoder - вполне универсальные варианты
#     - Закодировать все признаки методом OHE, а модель машины методом OE
#     - Заменить использование линейных моделей (так как их рассмотрение не обязательно) и использовать единственный метод кодировки OE.
# - Сделать 2 набора данных
#     - Закодировать для линейных моделей методом OHE, для остальных OE (или внутренний метод кодирования данных)
#     
# P.S. Отмечу, что encoder правильно применять после разбиения данных и обучать только на обучающей выборке, для остальных выборок просто использовать transform. Примеры использования с объяснениями можно найти посмотреь https://colab.research.google.com/drive/1_gAMXcQKoCShB_l8FNtYEejMnosm9mvt?usp=sharing 
# 
# И не забывай использовать параметр `handle_unknown`
#   
# </div>
# 

# In[ ]:


#pd.set_option('display.max_columns', 400)
#features


# In[ ]:


features_train_old, features_test_old, target_train_old, target_test_old = train_test_split(
    features, target, test_size = 0.4, random_state = 12345)


# In[ ]:


features_valid_old, features_test_old, target_valid_old, target_test_old = train_test_split(
    features_test_old, target_test_old, test_size = 0.5, random_state = 12345)


# In[ ]:


# Создание экземпляра класса BinaryEncoder
encoder = ce.BinaryEncoder(cols=['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'Repaired'])

# Преобразование тренировочных и тестовых данных
features_train = encoder.fit_transform(features_train_old)
features_valid = encoder.transform(features_valid_old)
features_test = encoder.transform(features_test_old)


# <div class="alert alert-block alert-warning">
# <b>Изменения:</b> Воспользовался BinaryEncoder. Сделал валидационную выборку, ранее была только обучающая и тестовая. 
# </div>

# 
# <div class="alert alert-block alert-success">
# <b>Успех[2]:</b> Есть контакт
# </div>

# In[ ]:


pd.set_option('display.max_columns', 50)
features_train


# In[ ]:


numeric = ['RegistrationYear', 'Power', 'Kilometer']

scaler = StandardScaler()
scaler.fit(features_train[numeric]) 
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
features_test[numeric] = scaler.transform(features_test[numeric])

# масштабируем количественные признаки, которые затем будут использованы в моделях градиентного бустинга, так как они имеют свои
# внутренние методы кодировки

features_train_old[numeric] = scaler.transform(features_train_old[numeric])
features_valid_old[numeric] = scaler.transform(features_valid_old[numeric])
features_test_old[numeric] = scaler.transform(features_test_old[numeric])

print(features_valid.head())


# <div class="alert alert-block alert-success">
# <b>Успех:</b>  Масштабирование проведено корректно

# **Промежуточный вывод**
# 
# Категориальные переменные были преобразованы в числовые значения методом BinaryEncoder, количественные признаки были масштабированы. Выборка была поделена на обучающую, валидационную и тестовую.

# ## Обучение моделей

# Обучим разные модели, одна из которых — LightGBM, как минимум одна — не бустинг. Для каждой модели попробуем разные гиперпараметры.

# ### Дерево решений

# In[ ]:


best_model_tree = None
best_result = 5000000
max_depth = None
scores = []

for depth in range(1,11,3):
    model = DecisionTreeRegressor(max_depth = depth, random_state = 12345)
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    scores = cross_val_score(model, features_train, target_train_old, cv = 5, scoring=scoring)
    final_score = abs(np.mean(scores))       
    rmse = final_score ** 0.5
    result = rmse
    if result < best_result:
        best_result = result
        best_model_tree = model
        max_depth = depth        

print("RMSE на обучающей выборке:", best_result, 'и максимальную глубину:', max_depth)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nbest_model_tree.fit(features_train, target_train_old)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\npredict_valid_tree = best_model_tree.predict(features_valid)')


# In[ ]:


mse = mean_squared_error(target_valid_old, predict_valid_tree)
rmse = mse ** 0.5

print("RMSE на валидационной выборке:", rmse)


# ### Случайный лес

# In[ ]:


best_model_forest = None
best_result = 5000000
max_depth = None
n_estimators = None
scores = []

for depth in range(1,10,3):
     for est in range(20,40,20):
        model = RandomForestRegressor(max_depth = depth, n_estimators = est, random_state = 12345)
        scores = cross_val_score(model, features_train, target_train_old, cv = 5, scoring=scoring)
        final_score = abs(np.mean(scores))       
        rmse = final_score ** 0.5
        result = rmse
        if result < best_result:
            best_result = result
            best_model_forest = model
            max_depth = depth
            n_estimators = est

print("RMSE на обучающей выборке:", best_result,',','число деревьев:', n_estimators,' и максимальную глубину:', max_depth)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nbest_model_forest.fit(features_train, target_train_old)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\npredict_valid_forest = best_model_forest.predict(features_valid)')


# In[ ]:


mse = mean_squared_error(target_valid_old, predict_valid_forest)
rmse = mse ** 0.5

print("RMSE на валидационной выборке:", rmse)


# ### CatBoost

# При использовании CatBoost мы не должны пользоваться one-hot кодированием, поскольку это влияет на скорость обучения и на качество прогнозов. Вместо этого зададим категориальные признаки с помощью параметра cat_features.

# <div class="alert alert-block alert-success">
# <b>Успех:</b> Верно, но с другими моделями аналогичный нюанс
# </div>

# In[ ]:


best_model_cat = None
best_result = 5000000
max_iter = None
scores = []

for iter in range(990,1001,10):
    cat_features = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'Repaired']
    model = CatBoostRegressor(iterations=iter, random_state=12345, cat_features=cat_features)
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    model.fit(features_train_old, target_train_old, verbose=100, plot=True)
    scores = cross_val_score(model, features_train_old, target_train_old, cv=2, scoring=scoring)
    final_score = abs(np.mean(scores))
    rmse = final_score ** 0.5
    result = rmse
    if result < best_result:
        best_result = result
        best_model_cat = model
        max_iter = iter
               
print("RMSE на обучающей выборке:", best_result,', число деревьев:', max_iter)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nbest_model_cat.fit(features_train_old, target_train_old, cat_features=cat_features, verbose = 0)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\npredict_valid_cat = best_model_cat.predict(features_valid_old)')


# In[ ]:


mse = mean_squared_error(target_valid_old, predict_valid_cat)
rmse = mse ** 0.5

print("RMSE на валидационной выборке:", rmse)


# ### LGBMBoost

# 
# <div class="alert alert-block alert-info">
# <b>Совет:</b>  У LGBM тоже есть внутренний метод кодировки данных, который хорошо было бы попробовать
# </div>
# 

# In[ ]:


param_grid = {
    'max_depth': np.arange(1, 7, 3),
    'n_estimators': np.arange(20, 41, 20)
}

categorical_features = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'Repaired']

features_train_old[categorical_features] = features_train_old[categorical_features].astype('category')

model = LGBMRegressor(verbose=200, cat_features = categorical_features, random_state=12345)

random_search = RandomizedSearchCV(
    estimator=model, param_distributions=param_grid, cv=3, n_iter=10, scoring='neg_mean_squared_error', random_state=123)
random_search.fit(features_train_old, target_train_old)

best_params = random_search.best_params_
best_score = random_search.best_score_

best_model = LGBMRegressor(**best_params)

best_model.fit(features_train_old, target_train_old)

predict_train_LGBM_RS = best_model.predict(features_train_old)
mse = mean_squared_error(target_train_old, predict_train_LGBM_RS)
rmse = mse ** 0.5

print("RMSE на обучающей выборке:", rmse, ", Best Model Hyperparameters:", best_params)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nbest_model.fit(features_train_old, target_train_old)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfeatures_valid_old[categorical_features] = features_valid_old[categorical_features].astype('category')\n\npredict_valid_lgbm = best_model.predict(features_valid_old)")


# In[ ]:


mse = mean_squared_error(target_valid_old, predict_valid_lgbm)
rmse = mse ** 0.5

print("RMSE на валидационной выборке:", rmse)


# **Промежуточный вывод**
# 
# 1) DecisionTree: обучение 623 ms, предсказание 19.7 ms и RMSE на валидационной выборке: 2052.
# 
# 2) RandomForest: обучение 6.51 s, предсказания 87.8 ms и RMSE на валидационной выборке: 2191.
# 
# 3) CatBoost: обучение 2min 17s, предсказания 417 ms и RMSE на валидационной выборке: 1640.
# 
# 4) LGBMBoost: обучение 1.63 s, предсказание 128 ms и RMSE на валидационной выборке: 1860.
# 
# - качество предсказания:CatBoost, LGBMBoost, DecisionTree. 
# - время обучения модели: DecisionTree, LGBMBoost, RandomForest.  
# - время предсказания модели: DecisionTree, RandomForest, LGBMBoost
# 
# CatBoost обучается в разы медленнее, даже с испольщованием 2-кросс валидационных выборок (у LGBMBoost 3). Нельзя сказать, что разница с LGBMBoost в качестве незначительная, но все же, учитывая пожелания заказчика LGBMBoost предпочтительнее ввиду баланса времени обучения и качества предсказания.

# 
# <div class="alert alert-block alert-danger">
# <b>Ошибка:</b> Ты немного не правильно понял метрики времени: Нам нужно время обучения и время предсказания  вычислить  и проанализировать отдельно друг от друга
#     
# - время обучения это  чистый `.fit()` модели - без подбора гиперпараметров и без предсказаний
#     
# - время предсказания это только `.predict()` без обучения
# 
# p.s. можно вытаскивать  все  метрики  интересующие заказчика лаконично из GridSearchCV/RandomizedSearchCV, все они лежат в `.cv_results_`
# </div>
# 
# 

# <div class="alert alert-block alert-warning">
# <b>Изменения:</b> Поправил.
# </div>

# 
# <div class="alert alert-block alert-success">
# <b>Успех[2]:</b> Все метрики корректно вычислены и проанализированы, выбор модели обоснован
# </div>

# ## Анализ выбранной модели

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfeatures_test_old[categorical_features] = features_test_old[categorical_features].astype('category')\n\npredict_test_lgbm = best_model.predict(features_test_old)")


# In[ ]:


mse = mean_squared_error(target_test_old, predict_test_lgbm)
rmse = mse ** 0.5

print("RMSE на тестовой выборке:", rmse)


# In[ ]:


def importances(model,features):
    features=features.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 5))
    plt.title('Важность функции')
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Относительная важность')
    plt.show()


# In[ ]:


importances(best_model,features_test_old)


# 
# <div class="alert alert-block alert-success">
# <b>Успех[2]:</b> На тестовой выборке получено хорошее качество
# </div>

# **Промежуточный вывод**
# 
# Модель LGBMboost показала воспроизводимую метрику RMSE на тествой выборке: 1860. Наиболее важные признаки, влияющие на предсказание модели, это:
# - год регистрации автомобиля
# - модель
# - мощность.

# <div class="alert alert-block alert-danger">
#    
# <b>Ошибка:</b>
#    
# - Тестовую выборку мы не должны использовать при вычислении метрик для анализа. Она должна использоваться только для единственной лучшей модели после анализа  в шаге с тестированием.
# - У нас заданы 3 метрики интересующие заказчика: `время обучения: .fit()`, `скорость предсказания: .predict()` и `качество: RMSE`. Эти метрики мы получаем и анализируем отдельно друг от друга (не на тестовой выборке и без учета времени подбора  гиперпаметров) 
# - Получить их можем либо с помощью валидационной выборки либо используя кросс-валидационные методы (в GridSearchCV например есть все данные метрики в аттрибуте cv_results_).   В случае если не используется ни gridsearch ни валидационная выборка, то можно время предсказания замерить на тренировочной (качество так замерить нельзя).
# - Потом проводится анализ и выбирается ОДНА наилучшая модель.
# - И только затем для ОДНОЙ наилучшей модели  проводится тестирование.
#   
# Исправь пожалуйста это здесь и везде внизу
# </div>
# 

# <div class="alert alert-block alert-danger">
# <b>Ошибка:</b> После анализа и выбора одной наилучшей модели, должно идти  ее тестирование. Другие модели тестовую выборку использовать не должны.
# </div>
# 

# ## Общий вывод

# Были предоставлены данные, которые имели 15 колонок и 354369 строк. Были пропущенные и аномальные значения.
# 
# Столбцы NumberOfPictures, RegistrationMonth, PostalCode, DateCrawled, DateCreated, LastSeen были удалены, так как их значение на стоимость автомобиля, вероятно, незначимо. Также были удаления строки, где не была указана модель, так как восстановить признак было бы невозможно логическим путем. 10642 наблюдений имели 0 в качестве цены, еще около 1000 цена была очень низкая. Учитывая, что это целевой признак, были сохранены только наблюдения с ценой выше 99 евро. Также были удалены наблюдения с годом регистрации автомобиля позже 2023 и раньше 1950 гг. 
# 
# Часть пропусков в столбце Repaired (автомобили, которые имели ремонт), были заполнены с использованием допущения, что большинство машин, согласно заполненным данным, выпущенных до 2006 г. и с пробегом более 150000 км. имели ремонт. 366 значений в столбце Power (мощность) имели значения, которые мало соответствовали раельности. Они были заменены на медианные значения мощности в соответствии с моделью и брендом. VehicleType (тип кузова), FuelType (вид топлива), Gearbox (тип коробки передач), часть Repaired (ремонт) были приблизетельно восстановлены по моде (самым частым значениям) для своей модель и марки. В столбце вид топлива petrol и gasoline (бензин) и lpg и cng (газ) были объединены.
# 
# Категориальные переменные были преобразованы в числовые значения методом BinaryEncoder, количественные признаки были масштабированы. Выборка была поделена на обучающую, валидационную и тестовую. 
# 
# При предсказании на валидационное выборке CatBoost имела самый низкий rmse 1640, затраченное на обучение составило 2min 17s. 
# 
# В целом, модели можно было разделить по параметрам от лучшего к худшему в следующем порядке:
# 
# - качество предсказания:CatBoost, LGBMBoost, DecisionTree.
# - время обучения модели: DecisionTree, LGBMBoost, RandomForest.
# - время предсказания модели: DecisionTree, RandomForest, LGBMBoost
# 
# CatBoost обучается в раза медленнее. Нельзя сказать, что разница с LGBMBoost в качестве незначительная (1640 против 1860), но все же, учитывая пожелания заказчика LGBMBoost предпочтительнее ввиду баланса времени обучения и качества предсказания (2min 31s против 1.7s).
# 
# Модель LGBMboost показала воспроизводимую метрику RMSE на тествой выборке: 1860. Наиболее важные признаки, влияющие на предсказание модели, это:
# - год регистрации автомобиля
# - модель
# - мощность.
# 
# Учитывая важность этих признаков:
# 1) Обратить внимание. Последняя дата скачивания анкеты из базы 2016-04-07 14:36:58, но пристутствуют автомобили от 2017 г., что составляет 11844 наблюдений. Интерпретировать данные изменения лучше всего с руководителями сервиса «Не бит, не крашен», возможно, что дата ошибочная. Скачали базу после обновления, например. В ходе предсказания данные от 2017 до 2023 г. были сохранены. 
# 
# 2) 30321 автомобиля имели мощность 0. Было взято допущение, что это машины без двигателя, так как медианная цена на них в 2 раза ниже. Все же стоит посоветоваться с экспертами рынка, чтобы узнать, является ли это аномалией (около 10% база данных). 

# 
# <div class="alert alert-block alert-info">
# <b>Совет:</b> 
# 
# Также если говорить, что можно ещё улучшить в подобных проектах, то я бы выделил такие моменты:<br>
#     
# 1) Напомню, что для понимания, а какие в итоге факторы важны при моделировании, можно выводить их важность, использую feature_importances_
#     
# 2) У нас разный возраст машин. Есть гипотеза, что для разных возрастов - своё ценообразование. Поэтому, можно попробовать ввести фактор "тип возраста" (ретро, супер-ретро, старая, новая... надо подумать..)..<br>
# 
# 
# </div>
# 

# ## Чек-лист проверки

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  Jupyter Notebook открыт
# - [х]  Весь код выполняется без ошибок
# - [х]  Ячейки с кодом расположены в порядке исполнения
# - [х]  Выполнена загрузка и подготовка данных
# - [х]  Выполнено обучение моделей
# - [х]  Есть анализ скорости работы и качества моделей

# In[ ]:




