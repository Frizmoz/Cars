import pandas as pd
import gradio as gr
from gradio.components import File,Dataframe
import pickle

# Загрузить данные и вывести информацию о датасете
def load_data(file,model_weights):
    df = pd.read_csv(file.name,low_memory=False)
    
    # Чистка
    thresh = int(df.shape[0] * 0.01)
    columns = df.columns[df.isna().sum() <= thresh]
    df = df[columns]
    df.dropna(inplace=True)
    
    # # Обработка пропущенных значений (замена на среднее значение)
    # df.fillna(df.mean(), inplace=True)
    
    # Удалям лишние столбцы, котрые были удален при тренеровки
    X = df.drop(columns=['target','report_date','col1454','client_id','col2183', 'col2181', 'col2176',
       'col2175', 'col2184', 'col2167', 'col2174', 'col2173', 'col2189',
       'col2168', 'col1453'],axis=1)

    # Инициализируйте и обучите модель случайного леса
    # load_model = pickle.load(open(model_weights, 'rb'))
    load_model = pickle.load(open(model_weights.name, 'rb'))

    # Получите вероятности классов для каждой строки
    probabilities = load_model.predict_proba(X)

    # Добавьте столбец с вероятностями в DataFrame
    df['probability'] = probabilities[:, 1]  # Используйте вероятности класса 1
    
    d = pd.DataFrame()
    d['client_id']=df['client_id']
    d['probability']=df['probability']
    # Верните DataFrame с добавленным столбцом вероятностей
    outputs_csv='output.csv'
    
    return df.to_csv('submission_file.csv',index= False)

# Определить интерфейс пользователя
def interface():
    file = [File(label="Выберите CSV файл для загрузки",file_types = ['.csv']),
           File(label="Выберите веса")]
    return gr.Interface(fn=load_data, inputs=file, outputs=None, title="Предсказание вероятности покупки")

# Запустить интерфейс пользователя
if __name__ == '__main__':
    app = interface()
    app.launch()