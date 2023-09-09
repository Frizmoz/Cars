import pandas as pd
import gradio as gr
import pickle
from gradio.components import File,Dataframe

# Загрузить данные и вывести информацию о датасете
def load_data(file,model_weights):
    df = pd.read_csv(file.name,low_memory=False)

    #Вычислите процент пропущенных данных для каждого столбца
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    #Выберите столбцы, в которых пропущено менее или равно 1% данных
    columns_to_keep = missing_percentage[missing_percentage >= 1].index
    #Выведите названия столбцов, которые соответствуют вашему критерию
    df.drop(columns=columns_to_keep,inplace=True)
    df.dropna(inplace=True)
    
    # Удалям лишние столбцы, котрые были удален при тренеровки
    X = df.drop(columns=['target','report_date','col1454','client_id','col2183', 'col2181', 'col2176',
       'col2175', 'col2184', 'col2167', 'col2174', 'col2173', 'col2189',
       'col2168', 'col1453'],axis=1)

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