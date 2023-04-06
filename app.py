from flask import Flask, request, jsonify
import pandas as pd
import json
app = Flask(__name__)

from main2 import getPredictForAllDaysInNextWeek
from flask_cors import CORS, cross_origin

CORS(app)


@app.route('/api/ma-route', methods=['get'])
def post_route():
    result = getPredictForAllDaysInNextWeek()
    df_filtre = result[['Date_Heure', 'consommation', 'consommation_pred']]
    df_filtre_json = df_filtre.to_json(orient='records')
    return jsonify(data=json.loads(df_filtre_json))


if __name__ == "__main__":
    app.run(debug=True)
