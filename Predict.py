import pandas as pd
import gradio as gr
import pickle
from gradio.components import File,Dataframe

# Загрузить данные и вывести информацию о датасете
def load_data(file,model_weights):
    df = pd.read_csv(file.name,low_memory=False)
    
    # Отсвляем только те столбцы на котрых тренеровали модель ранее 
    df = df[['client_id','col2169', 'col2170', 'col2171', 'col2172', 'col2177', 'col2178',
           'col2179', 'col2180', 'col2182', 'col2185', 'col2186', 'col2187',
           'col2188', 'col2190', 'col2220', 'col2221', 'col2222', 'col2292',
           'col2293', 'col2294', 'col2316', 'col2317', 'col2318', 'col2340',
           'col2341', 'col2342', 'col2364', 'col2365', 'col2366', 'col2388',
           'col2389', 'col2390', 'col2663']]
    df.dropna(inplace=True)
    X=df.drop(columns=['client_id'],axis=1)

    # Инициализируйте и обучите модель случайного леса
    load_model = pickle.load(open(model_weights.name, 'rb'))

    # Получите вероятности классов для каждой строки
    probabilities = load_model.predict_proba(X)
    # Добавьте столбец с вероятностями в DataFrame
    df['score'] = probabilities[:, 1]  # Используйте вероятности класса 1

        # Полный датасет с вероятостяи
    d = pd.DataFrame()
    d['client_id']=df['client_id']
    d['score']=df['score']
    df.to_csv('submission_file.csv',index= False)

    # Датасет целевой аудитории 
    dff=pd.DataFrame()
    dff['client_id']=d['client_id']
    dff['score']=d['score']
    dff.drop_duplicates(inplace=True)
    dff=dff[dff['score']>=0.5] # порог удаления прользователей
    dff.sort_values(by=['score'],ascending=False,inplace=True)
    dff.to_csv('submission_GOOD_file.csv',index= False)
    return dff

# Определить интерфейс пользователя
def interface():
    file = [File(label="Выберите CSV файл для загрузки",file_types = ['.csv']),
           File(label="Выберите веса модели")]
    outputs=Dataframe(label="Пользователи")
    return gr.Interface(fn=load_data, inputs=file, outputs=outputs, title="Предсказание вероятности покупки")

# Запустить интерфейс пользователя
if __name__ == '__main__':
    app = interface()
    app.launch()