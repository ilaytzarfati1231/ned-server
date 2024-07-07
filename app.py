from flask import Flask, request, send_file, jsonify
import os
import logging

from flask_cors import CORS
from core import  generate_image_from_automata, hash_regex_string, parse_regex_string, automata_edit_distance_graph,inf_inf

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

saved_product = {}
saved_product_inf_inf = {}

@app.route('/generate-automata-image', methods=['POST'])
def generate_image_endpoint():
    data = request.json
    regex_string = data.get('regex_string')
    if not regex_string:
        logger.error('No regex string provided')
        return jsonify({'error': 'No regex string provided'}), 400
    
   
    hash_value = hash_regex_string(regex_string)
    cached_image_path = os.path.join(CACHE_DIR, f'{hash_value}')

    if not os.path.exists(cached_image_path + '.png'):
        logger.info(f'Generating image for regex: {regex_string}')
        automata = None
        if "is edit distance automaton" in regex_string:
            regex_string = regex_string.split("is edit distance automaton")[0]
            regex_string_first=regex_string.split("$")[0]
            regex_string_second=regex_string.split("$")[1]
            if (regex_string_first,regex_string_second) in saved_product:
                automata = saved_product[(regex_string_first,regex_string_second)]
            else:
                first_automata = parse_regex_string(regex_string_first)
                second_automata = parse_regex_string(regex_string_second)
                automata = automata_edit_distance_graph(first_automata, second_automata)
                saved_product[(regex_string_first,regex_string_second)] = automata
        else:
            automata = parse_regex_string(regex_string)
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
    regex_strings = [item.get('regex_string') for item in data]
    if not regex_strings:
        logger.error('No regex strings provided')
        return jsonify({'error': 'No regex strings provided'}), 400
    regex_string = regex_strings[0]
    if "is edit distance automaton" in regex_string:
            regex_string = regex_string.split("is edit distance automaton")[0]
            regex_string_first=regex_string.split("$")[0]
            regex_string_second=regex_string.split("$")[1]
            automata = None
            if (regex_string_first,regex_string_second) in saved_product_inf_inf:
                return jsonify({'inf_inf': saved_product_inf_inf[(regex_string_first,regex_string_second)]})
            if (regex_string_first,regex_string_second) in saved_product:
                automata = saved_product[(regex_string_first,regex_string_second)]
            else:
                first_automata = parse_regex_string(regex_string_first)
                second_automata = parse_regex_string(regex_string_second)
                automata = automata_edit_distance_graph(first_automata, second_automata)
                saved_product[(regex_string_first,regex_string_second)] = automata
            inf_inf_value = inf_inf(automata)
            saved_product_inf_inf[(regex_string_first,regex_string_second)] = inf_inf_value
            return jsonify({'inf_inf': inf_inf_value})
    else:
        return jsonify({'error': 'No regex strings provided'}), 400


if __name__ == '__main__':
    app.run(debug=True)
