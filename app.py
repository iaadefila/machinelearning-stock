

from flask import Flask, render_template, request, jsonify, make_response
from utilis import*
from keras.models import load_model

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/model", methods=["POST"])
def model():
        req = request.get_json()
        print(req)
        # Call the respected model here and provide the variables with the values
        dataset_url = req['location']
        data_name = req['symbol']
        
        if(data_name == 'catt'):
            #Load Models 
            model_lstm = load_model('MV-LSTM_dataset1_hyperband.h5', compile=False)
            model_saramix = SARIMAXResults.load('model1.pkl')
            model_conv = load_model('ConvLSTM2D_dataset1.h5', compile=False)
            
            # preprocess dataset and Prepare Dataframe 
            df = preprocess(dataset_url)
            
            #Get predictions from the different models
            lstm_val = LSTM_pred(df, model_lstm)
            saramix_val = SARIMAX_pred(df, model_saramix)
            conv_val = ConvLstm_pred(df, model_conv)
            
        elif(data_name == 'qfls'):
            #Load Models 
            model_lstm = load_model('MV-LSTM_dataset2_hyperband.h5', compile=False)
            model_saramix = SARIMAXResults.load('model2.pkl')
            model_conv = load_model('ConvLSTM2D_dataset2.h5', compile=False)
            
            # preprocess dataset and Prepare Dataframe 
            df = preprocess(dataset_url)
            
            #Get predictions from the different models
            lstm_val = LSTM_pred(df, model_lstm)
            saramix_val = SARIMAX_pred(df, model_saramix)
            conv_val = ConvLstm_pred(df, model_conv)
            
        elif(data_name == '8030'):
            #Load Models 
            model_lstm = load_model('MV-LSTM_dataset3_hyperband.h5', compile=False)
            model_saramix = SARIMAXResults.load('model3.pkl')
            model_conv = load_model('ConvLSTM2D_dataset3.h5', compile=False)
            
            # preprocess dataset and Prepare Dataframe 
            df = preprocess(dataset_url)
            
            #Get predictions from the different models
            lstm_val = LSTM_pred(df, model_lstm)
            saramix_val = SARIMAX_pred(df, model_saramix)
            conv_val = ConvLstm_pred(df, model_conv)
            
        elif(data_name == 'rakprop'):
            #Load Models
            model_lstm = load_model('MV-LSTM_dataset4_hyperband.h5', compile=False)
            model_saramix = SARIMAXResults.load('model4.pkl')
            model_conv = load_model('ConvLSTM2D_dataset3.h5', compile=False)
            
            # preprocess dataset and Prepare Dataframe 
            df = preprocess(dataset_url)
            
            #Get predictions from the different models
            lstm_val = LSTM_pred(df, model_lstm)
            saramix_val = SARIMAX_pred(df, model_saramix)
            conv_val = ConvLstm_pred(df, model_conv)
            
        current = cp(dataset_url)
        arima = saramix_val
        lstm = lstm_val
        convlstm = conv_val
        res = make_response(jsonify({"convlstm":convlstm,"lstm":lstm,"arima":arima,"current":current}), 200)
        return res

if __name__ == '__main__':
        app.run(debug=True)
        
