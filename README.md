![Статус проекта: Почти Закончен](https://raw.githubusercontent.com/chftm/brand/main/Project%20Status/almost_finished.svg)

# probability_predictor
## О продукте 🌐

Продукт был сделан для решения задачи от [Самолет](https://samolet.ru/) хакатона "[Цифровой прорыв](https://hacks-ai.ru/)". Он предназначен для предсказания склонности клиента к приобретению машиноместа. Задачей было посчитать вероятность приобретения клиентом машиноместа в следующие 3 месяца.

## Использованные ЯП и библиотеки 📚

- Python:
	- [pandas](https://pandas.pydata.org)
	- [seaborn](https://seaborn.pydata.org)
	- [numpy](https://numpy.org)
	- [matplotlib](https://matplotlib.org)
	- [sklearn](https://scikit-learn.org)
	- [catboost](https://catboost.ai)
	- [gradio](https://www.gradio.app)


## Для запуска нужно ✨

- Установить [Python](https://python.org/)
- Установить библиотеки:
    `pip install pandas seaborn numpy matplotlib sklearn catboost gradio`
- Скачать репозиторий проекта по [ссылке](https://github.com/Frizmoz/Cars), используя команду: git clone https://github.com/Frizmoz/Cars
- Запустить скрипт

## Как работает приложение? ⚙

После клонирования репозитория запустите скрипт script.py. В консоли вам будет выдана ссылка, перейдите по ней.


![ui-sta | 450](screens\Screenshot_10.png)

Далее вам необходимо загрузить по одному файлу в каждую из двух областей:
1. В первую область загрузите свой **csv** файл.

![ui-folder | 450](screens\Screenshot_4.png)

2. Во вторую область загрузите файл **RandomForest.sav** из репозитория.

![ui-2 | 450](screens\Screenshot_5.png)

После  этого нажмите на кнопку **Submit**.

![ui-3 | 450](screens\Screenshot_6.png)

Ожидайте до появления файла **submission_file.csv** в папке репозитория. В нем записаны данные о каждом клиенте, где последний стобец показывает вероятность покупки машиноместа клиентом в ближайщие 3 месяца.

## Принцип работы ⚒

1.  Запуск файла скрипта(script.py).
2.  Загрука файлов: датасет csv и RandomForest.sav.
3.  Происходит предобработка данных из csv файла.
4.  Вычисляется вероятность покупки машиноместа клиентами с помощью модели Random Forest.
5.  Выдаётся csv файл(submission_file.csv) с вероятностями для каждого клиента.

## Создатели ❤
 
[Frizmoz](https://github.com/Frizmoz) - аналитика данных, создание и обучение моделей машинного обучения

[EgorKA027](https://github.com/EgorKA027) - курирование проекта, обработка данных, создание презентации

[RalinaT18](https://github.com/RalinaT18) - помощь в дизайне

[ICoryphaeusI](https://github.com/ICoryphaeusI) - помощь в дизайне, разработка вспомогательных файлов
