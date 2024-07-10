from flask import Flask, request, send_file, jsonify
import os
import logging

from flask_cors import CORS
from core import  generate_image_from_automata, hash_regex_string, parse_regex_string, automata_edit_distance_graph,inf_inf,sum_inf_inf,parse_dict_to_automata

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

saved_product = {}
saved_product_inf_inf = {}
saved_product_Sum_inf_inf = {}
saved_automata = {}
@app.route('/generate-automata-image', methods=['POST'])
def generate_image_endpoint():
    data = request.json
    name = data.get('name')
    hash_value = hash_regex_string(name)
    cached_image_path = os.path.join(CACHE_DIR, f'{hash_value}')
    if not os.path.exists(cached_image_path + '.png'):
        logger.info(f'Generating image for regex: {name}')
        if "is edit distance automaton" in  name:
                regex_string = name.split("is edit distance automaton")[0]
                regex_string_first=regex_string.split("$")[0]
                regex_string_second=regex_string.split("$")[1]
                if (regex_string_first,regex_string_second) in saved_product:
                    automata = saved_product[(regex_string_first,regex_string_second)]
                else:
                    first_automata,second_automata = _convert_to_automata(data)
                    automata = automata_edit_distance_graph(first_automata, second_automata)
                    saved_product[(regex_string_first,regex_string_second)] = automata
        else:
            automata = _convert_to_automata(data)[0]
            saved_automata[name] = automata
        generate_image_from_automata(automata, cached_image_path)
    else:
        logger.info(f'Using cached image for regex: {regex_string}')
    cached_image_path += '.png'
    response = send_file(cached_image_path, mimetype='image/png')
    response.headers['Cache-Control'] = 'no-store'
    return response
    
@app.route('/Inf_Inf', methods=['POST'])
def Inf_Inf():
    data = request.json
    name = data.get('name')
    if "is edit distance automaton" in  name:
        first_automata,second_automata = _convert_to_automata(data)
        if first_automata is None or second_automata is None:
            return jsonify({'error': 'No automata provided'}), 400
        automata = automata_edit_distance_graph(first_automata, second_automata)
        first_name = name.split("$")[0]
        second_name = name.split("$")[1]
        saved_product[(first_name,second_name)] = automata
        inf_inf_value = inf_inf(automata)
        saved_product_inf_inf[(first_name,second_name)] = inf_inf_value
        return jsonify({'inf_inf': inf_inf_value})
    return jsonify({'error': 'No regex strings provided or names'}), 400
        
@app.route('/Sum_Inf_Inf', methods=['POST'])
def Sum_Inf_Inf():
    data = request.json
    name = data.get('name')
    if "is edit distance automaton" in  name:
        first_automata,second_automata = _convert_to_automata(data)
        if first_automata is None or second_automata is None:
            return jsonify({'error': 'No automata provided'}), 400
        automata = automata_edit_distance_graph(first_automata, second_automata)
        first_name = name.split("$")[0]
        second_name = name.split("$")[1]
        saved_product[(first_name,second_name)] = automata
        sum_inf_inf_value = sum_inf_inf(automata)
        saved_product_Sum_inf_inf[(first_name,second_name)] = sum_inf_inf_value
        return jsonify({'Sum_inf_inf': sum_inf_inf_value})
    return jsonify({'error': 'No regex strings provided or names'}), 400
    
def _convert_to_automata(data):
    name = data.get('name')
    first_automata = None
    second_automata = None
    if "is edit distance automaton" in name:
        name = name.split("is edit distance automaton")[0]
        name_first=name.split("$")[0]
        name_second=name.split("$")[1]
        Q =data.get('Q')
        Q1,Q2 = convert_from_SEP(Q)
        Sigma = data.get('Sigma')
        Sigma1,Sigma2 = convert_from_SEP(Sigma)
        Delta = data.get('Delta')
        Delta1,Delta2 = convert_from_SEP(Delta)
        q0 = data.get('q0').split("SEP")
        q01 = q0[0].replace(" ","")
        q02 = q0[1].replace(" ","")
        F = data.get('F')
        F1,F2 = convert_from_SEP(F)
        if len(Q1)>0:
            first_automata = saved_automata.get(name_first,parse_dict_to_automata(Q1,Sigma1,Delta1,q01,F1))
        else:
            first_automata = saved_automata.get(name_first,parse_regex_string(name_first))
        saved_automata[name_first] = first_automata
        if len(Q2)>0:
            second_automata = saved_automata.get(name_second,parse_dict_to_automata(Q2,Sigma2,Delta2,q02,F2))
        else:
            second_automata = saved_automata.get(name_second,parse_regex_string(name_second))
        saved_automata[name_second] = second_automata
    elif len(data.get('Q')) > 0:
        Q = data.get('Q')
        Sigma = data.get('Sigma')
        Delta = data.get('Delta')
        q0 = data.get('q0')
        F = data.get('F')
        first_automata = saved_automata.get(name,parse_dict_to_automata(Q,Sigma,Delta,q0,F))
        saved_automata[name] = first_automata
    else:
        first_automata = saved_automata.get(name,parse_regex_string(name))
        saved_automata[name] = first_automata
    return first_automata,second_automata




def convert_from_SEP(S):
    seen = False
    S1 = []
    S2 = []
    for s in S:
        if s == "SEP":
            seen = True
            continue
        elif not seen:
            S1.append(s)
        else:
            S2.append(s)
    return S1,S2


if __name__ == '__main__':
    app.run(debug=True)
