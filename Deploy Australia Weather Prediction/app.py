from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('AusWeather.pkl','rb'))

app = Flask(__name__)

dfloc = {'Adelaide': 0.21766666666666667, 'Albury': 0.20516258799865908, 'AliceSprings': 0.07896505376344086, 'BadgerysCreek': 0.19797486033519554, 'Ballarat': 0.2578490313961256, 'Bendigo': 0.18603874415497662, 'Brisbane': 0.22532051282051282, 'Cairns': 0.3153428377460964, 'Canberra': 0.18116883116883117, 'Cobar': 0.12844036697247707, 'CoffsHarbour': 0.29711246200607905, 'Dartmoor': 0.31243611584327086, 'Darwin': 0.2640610104861773, 'GoldCoast': 0.2600349040139616, 'Hobart': 0.238621997471555, 'Katherine': 0.16403162055335968, 'Launceston': 0.22842809364548494, 'Melbourne': 0.22319060250094733, 'Mildura': 0.10896367877374209, 'Moree': 0.12463235294117647, 'MountGambier': 0.3, 'MountGinini': 0.2826169646253328, 'Nhil': 0.1548304542546385, 'NorahHead': 0.27565880721220526, 'NorfolkIsland': 0.3081695966907963, 'Nuriootpa': 0.1951219512195122, 'PearceRAAF': 0.16070726915520628, 'Penrith': 0.19896729776247848, 'Perth': 0.1955779548040969, 'Portland': 0.3672297297297297, 'Richmond': 0.18920788654444828, 'Sale': 0.21111499475707796, 'SalmonGums': 0.16, 'Sydney': 0.2576710501238803, 'Townsville': 0.16971713810316139, 'Tuggeranong': 0.18962008141112618, 'Uluru': 0.07640297498309669, 'WaggaWagga': 0.17679180887372015, 'Walpole': 0.33590308370044053, 'Watsonia': 0.24552516041877745, 'Williamtown': 0.23786407766990292, 'Witchcliffe': 0.2973901098901099, 'Wollongong': 0.23960463531015677, 'Woomera': 0.06687033265444671}
dfdir = {'E': 0.14657762938230384, 'ENE': 0.15901060070671377, 'ESE': 0.16151297625621203, 'N': 0.26865003914550944, 'NE': 0.18494715795487005, 'NNE': 0.22865662272441933, 'NNW': 0.2822915066810014, 'NW': 0.2826552462526767, 'S': 0.22203408962636867, 'SE': 0.1850449085596797, 'SSE': 0.1922515440763616, 'SSW': 0.21789017679428638, 'SW': 0.20075844633417606, 'W': 0.2651608910891089, 'WNW': 0.2791457286432161, 'WSW': 0.2308566234946603}
dftoday = {'No': 0.14960652956147794, 'Yes': 0.4630438521066208}

@app.route('/')
def main_page():
    return render_template('weather.html')

@app.route('/predict',methods=['Post','Get'])
def predict():  
    Location = request.form['Location']
    Rainfall = request.form['Rainfall']
    WindGustDir = request.form['WindGustDirection']
    WindGustSpeed = request.form['WindGustSpeed']
    RainToday = request.form['RainToday']
    Humidity = request.form['Humidity']
    Pressure = request.form['Pressure']
    Temp = request.form['Temperature']
    arr = np.array([[Location,Rainfall,WindGustDir,WindGustSpeed,RainToday,Humidity,Pressure,Temp]])
    arr = pd.DataFrame(arr,columns=(['Location', 'Rainfall', 'WindGustDir', 'WindGustSpeed', 'RainToday','Humidity', 'Pressure', 'Temp']))
    arr['Location'] = arr['Location'].map(dfloc)
    arr['WindGustDir'] = arr['WindGustDir'].map(dfdir)
    arr['RainToday'] = arr['RainToday'].map(dftoday) 
    for i in arr.columns:  
        arr[i] = arr[i].astype('float64')
    prediction = model.predict_proba(arr)[:,1:]
    if prediction>0.5:
        return render_template('weather.html',pred='There\'s a very high probability of Rain Tomorrow.\nThe Chance of Raining Tomorrow is {}%'.format(round((prediction[0][0]*100),2)))
    else:
        return render_template('weather.html',pred='There\'s very less probabilty of Rain Tomorrow.\n The Chance of Raining Tomorrow is {}%'.format(round((prediction[0][0]*100),2)))
    

if __name__ == "__main__":
    app.run(debug=True)
